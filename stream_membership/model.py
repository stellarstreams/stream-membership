"""
TODO:
- Move plotting and grid evaluation stuff to the new ModelComponent class
- Add support for ModelComponent to be combined into a mixture. All components must
  have the same coordinate names, unique names
"""


import abc
import copy
import inspect
from functools import partial
from typing import Any

import equinox as eqx
import jax.numpy as jnp
import numpy as np
import numpyro
import numpyro.distributions as dist
from jax.scipy.special import logsumexp
from jax.typing import ArrayLike
from jax_ext.integrate import ln_simpson

from .plot import _plot_projections
from .utils import del_in_nested_dict, get_from_nested_dict, set_in_nested_dict


class ModelComponent(eqx.Module):
    name: str
    coord_distributions: dict[str | tuple, Any]
    coord_parameters: dict[
        str | tuple, dict[str, dist.Distribution | tuple | ArrayLike | dict]
    ]

    def _make_numpyro_name(
        self, coord_name: str | tuple[str, str], arg_name: str | None = None
    ) -> str:
        """
        Convert a nested set of component name (this class name), coordinate name, and
        parameter name into a single string for naming a parameter with
        numpyro.sample().

        Parameters
        ----------
        coord_name
            The name of the coordinate in the component. If a coordinate can only be
            modeled as a joint, pass a tuple of strings.
        arg_name
            The name of the parameter used in the model component for the coordinate.
        """
        # TODO: need to validate somewhere that coordinate names can't have "-" and
        # component, coordinate, and parameter names can't have ":"
        if isinstance(coord_name, tuple):
            coord_name = "-".join(coord_name)

        name = f"{self.name}:{coord_name}"
        if arg_name is None:
            return name
        return f"{name}:{arg_name}"

    def _expand_numpyro_name(self, numpyro_name: str) -> tuple[str, str | tuple, str]:
        """
        Convert a numpyro name into a tuple of component name, coordinate name, and
        parameter name.

        Parameters
        ----------
        numpyro_name
            The name of the parameter in the numpyro model, i.e. the name of a parameter
            specified with numpyro.sample(). In the context of this model, this should
            be something like "background:phi2:loc", where the model is named
            "background", the coordinate is named "phi2", and the parameter is named
            "loc".
        """
        bits = numpyro_name.split(":")
        return (
            bits[0],
            tuple(bits[1].split("-")) if "-" in bits[1] else bits[1],
            bits[2],
        )

    def _expand_numpyro_params(self, pars: dict[str, Any]) -> dict[str | tuple, Any]:
        """
        Convert a dictionary of numpyro parameters into a nested dictionary where the keys
        are the component name, coordinate name, and parameter name.

        Parameters
        ----------
        pars
            A dictionary of numpyro parameters where the keys are the names of the
            parameters created with numpyro.sample().
        """
        expanded_pars = {}
        for k, v in pars.items():
            name, coord_name, arg_name = self._expand_numpyro_name(k)
            if name not in expanded_pars:
                expanded_pars[name] = {}
            if coord_name not in expanded_pars[name]:
                expanded_pars[name][coord_name] = {}
            expanded_pars[name][coord_name][arg_name] = v

        return expanded_pars

    def make_dists(self, pars: dict[str, Any] | None = None) -> dict[str | tuple, Any]:
        """
        Make a dictionary of distributions for each coordinate in the component.

        Parameters
        ----------
        pars
            A dictionary of parameters to pass to the numpyro.sample() calls that
            create the distributions. The dictionary should be structured as follows:
            {
                "coord_name": {
                    "arg_name": value
                }
            }
            where "coord_name" is the name of the coordinate and "arg_name" is the name
            of the argument to pass to the numpyro.sample() call that creates the
            distribution. "value" is the value to pass to the numpyro.sample() call.
        """
        if pars is None:
            pars = {}

        dists = {}
        for coord_name, Distribution in self.coord_distributions.items():
            kwargs = {}
            for arg, val in self.coord_parameters.get(coord_name, {}).items():
                numpyro_name = self._make_numpyro_name(coord_name, arg)

                # Note: passing in a tuple as a value is a way to wrap the value in a
                # function or outer distribution, for example for a mixture model
                if isinstance(val, tuple):
                    wrapper, val = val  # noqa: PLW2901
                else:
                    wrapper = lambda x: x  # noqa: E731

                if arg in pars.get(coord_name, {}):
                    # If an argument is passed in the pars dictionary, use that value.
                    # This is useful, for example, for constructing the coordinate
                    # distributions once a model is optimized or sampled, so you can
                    # pass in parameter values to evaluate the model.
                    par = pars[coord_name][arg]
                elif isinstance(val, dict):
                    par = numpyro.sample(numpyro_name, **val)
                elif isinstance(val, dist.Distribution):
                    par = numpyro.sample(numpyro_name, val)
                else:
                    par = val
                kwargs[arg] = wrapper(par)

            dists[coord_name] = Distribution(**kwargs)

        return dists

    def __call__(self, data: dict[str, ArrayLike]) -> None:
        """
        This sets up the model component in numpyro.
        """
        dists = self.make_dists()
        for coord_name, dist_ in dists.items():
            if isinstance(coord_name, tuple):
                _data = jnp.stack([data[k] for k in coord_name], axis=-1)
                _data_err = None  # TODO: we don't support errors for joint coordinates
            else:
                _data = data[coord_name]
                _data_err = (
                    data[f"{coord_name}_err"] if f"{coord_name}_err" in data else None
                )

            numpyro_name = self._make_numpyro_name(coord_name)
            if _data_err is not None:
                model_val = numpyro.sample(numpyro_name, dist_)
                numpyro.sample(
                    f"{numpyro_name}-obs",
                    dist.Normal(model_val, data[f"{coord_name}_err"]),
                    obs=_data,
                )
            else:
                numpyro.sample(f"{numpyro_name}-obs", dist_, obs=_data)


class ModelBase:
    @classmethod
    def optimize(cls, data, init_params, jaxopt_kwargs=None, use_bounds=True, **kwargs):
        """
        A wrapper around numpyro_ext.optim utilities, which enable jaxopt optimization
        for numpyro models.
        """
        import jaxopt

        if jaxopt_kwargs is None:
            jaxopt_kwargs = {}
        jaxopt_kwargs.setdefault("maxiter", 1024)  # TODO: TOTALLY ARBITRARY

        optimize_kwargs = kwargs
        if use_bounds:
            optimize_kwargs["bounds"] = cls._get_jaxopt_bounds()
            optimize_kwargs["bounds"] = (
                cls._normalize_variable_keys(optimize_kwargs["bounds"][0]),
                cls._normalize_variable_keys(optimize_kwargs["bounds"][1]),
            )
            Optimizer = jaxopt.LBFGSB
        else:
            jaxopt_kwargs.setdefault("method", "BFGS")
            Optimizer = jaxopt.ScipyMinimize

        optimizer = Optimizer(**jaxopt_kwargs, fun=cls._objective)
        opt_res = optimizer.run(
            init_params=cls._normalize_variable_keys(init_params),
            data=cls._normalize_variable_keys(data),
            **optimize_kwargs,
        )
        return cls._expand_variable_keys(opt_res.params), opt_res.state

    ###################################################################################
    # Evaluating on grids and plotting
    #
    def _get_grids_2d(self, grids_1d, grid_coord_names):
        if grids_1d is None:
            grids_1d = {}

        grids_2d = {}
        for name_pair in grid_coord_names:
            for name in name_pair:
                if name not in grids_1d and name not in self.default_grids:
                    msg = f"No default grid for {name}, so you must specify it via the `grids` argument"
                    raise ValueError(msg)
            grids_2d[name_pair] = np.meshgrid(
                *[
                    grids_1d.get(name, self.default_grids.get(name))
                    for name in name_pair
                ]
            )

        return grids_2d

    def evaluate_on_2d_grids(self, grids=None, grid_coord_names=None):
        """
        Evaluate the log-number density on a 2D grid of coordinates. This is useful for
        creating plots of the predicted model number density.

        TODO:
        - Note that Pass in bin edges, evaluate on bin centers
        - grid_coord_names should be a list like [('phi1', 'phi2'), ('phi1', 'pm1')]
        """

        # TODO: add as another optional argument to this function?
        # Default "x" coordinate is the 0th coordinate name:
        default_x_coord = self.coord_names[0]

        if grid_coord_names is None:
            grid_coord_names = [
                (default_x_coord, coord) for coord in self.coord_names[1:]
            ]

        # validate grid_coord_names
        for name_pair in grid_coord_names:
            for name in name_pair:
                if name not in self.coord_names:
                    msg = f"{name} is not a valid coordinate name"
                    raise ValueError(msg)

        grids_2d = self._get_grids_2d(grids, grid_coord_names)

        evals = {}

        for name_pair in grid_coord_names:
            grid1, grid2 = grids_2d[name_pair]

            # Passed in are the grid edges, but we evaluate at the grid cell centers:
            bin_area = np.abs(np.diff(grid1[0])[None] * np.diff(grid2[:, 0])[:, None])

            grid1_c = 0.5 * (grid1[:-1, :-1] + grid1[1:, 1:])
            grid2_c = 0.5 * (grid2[:-1, :-1] + grid2[1:, 1:])

            # Fill a data dict with zeros for all coordinates not being plotted
            tmp_data = {name_pair[0]: grid1_c.ravel(), name_pair[1]: grid2_c.ravel()}
            for tmp_name in self.coord_names:
                if tmp_name not in name_pair:
                    tmp_data[tmp_name] = jnp.zeros_like(tmp_data[name_pair[0]])

            # Evaluate the model on the grid
            ln_ps = self.variable_ln_prob_density(tmp_data)

            # If the x coord is in a joint, also compute the marginals for the x
            # component of the joint distribution:
            if default_x_coord in name_pair and default_x_coord not in ln_ps:
                for joint_name_pair in ln_ps:
                    if (
                        isinstance(joint_name_pair, tuple)
                        and default_x_coord in joint_name_pair
                    ):
                        break
                else:
                    msg = "x coordinate not found in valid joint distribution"
                    raise ValueError(msg)

                ln_p = ln_ps[joint_name_pair]
                ln_p = ln_p.reshape(bin_area.shape)

                # pick grid1 or grid2 based on index, but probably grid2...
                idx = 0 if joint_name_pair[1] == default_x_coord else 1
                grid = [grid2_c, grid1_c][idx]
                ln_p = ln_simpson(ln_p, grid, axis=idx)
                ln_p = jnp.repeat(
                    jnp.expand_dims(ln_p, axis=idx), grid1_c.shape[idx], axis=idx
                ).ravel()
                ln_ps[default_x_coord] = ln_p - np.log(
                    grid.max(axis=idx)[0] - grid.min(axis=idx)[0]
                )

            if name_pair in self._joint_names:
                ln_p = ln_ps[name_pair]

            else:
                ln_p = ln_ps[name_pair[0]] + ln_ps[name_pair[1]]

            ln_n = self._pars["ln_N"] + ln_p.reshape(bin_area.shape) + np.log(bin_area)
            evals[name_pair] = ln_n

        return grids_2d, evals

    def plot_model_projections(
        self,
        grids=None,
        grid_coord_names=None,
        axes=None,
        label=True,
        pcolormesh_kwargs=None,
    ):
        """
        TODO:
        - grids are names like phi1, phi2, etc.
        - grid_coord_names are tuples like [(phi1, phi2), ...]
        """

        if grid_coord_names is None:
            grid_coord_names = [
                (self.coord_names[0], name) for name in self.coord_names[1:]
            ]

        grids, ln_ns = self.evaluate_on_2d_grids(
            grids=grids, grid_coord_names=grid_coord_names
        )

        ims = {k: np.exp(v) for k, v in ln_ns.items()}
        return _plot_projections(
            grids=grids,
            ims=ims,
            axes=axes,
            label=label,
            pcolormesh_kwargs=pcolormesh_kwargs,
        )

    def plot_residual_projections(
        self,
        data,
        grids=None,
        grid_coord_names=None,
        axes=None,
        label=True,
        smooth=1.0,
        pcolormesh_kwargs=None,
    ):
        """
        TODO:
        - grids are names like phi1, phi2, etc.
        - grid_coord_names are tuples like [(phi1, phi2), ...]
        """
        from scipy.ndimage import gaussian_filter

        if grid_coord_names is None:
            grid_coord_names = [
                (self.coord_names[0], name) for name in self.coord_names[1:]
            ]

        if grids is None:
            grids = {}
        coord_names = [name for name_pair in grid_coord_names for name in name_pair]
        grids_1d = {
            name: grids.get(name, self.default_grids.get(name)) for name in coord_names
        }

        # Evaluate the model at the grid midpoints
        im_grids, ln_ns = self.evaluate_on_2d_grids(
            grids=grids, grid_coord_names=grid_coord_names
        )

        model_ims = {k: np.exp(v) for k, v in ln_ns.items()}

        resid_ims = {}
        for name_pair, model_im in model_ims.items():
            # get the number density: density=True is the prob density, so need to
            # multiply back in the total number of data points
            H_data, *_ = np.histogram2d(
                data[name_pair[0]],
                data[name_pair[1]],
                bins=(grids_1d[name_pair[0]], grids_1d[name_pair[1]]),
            )
            data_im = H_data.T

            resid = model_im - data_im
            resid_ims[name_pair] = resid

            if smooth is not None:
                resid_ims[name_pair] = gaussian_filter(resid_ims[name_pair], smooth)

        if pcolormesh_kwargs is None:
            pcolormesh_kwargs = {}
        pcolormesh_kwargs.setdefault("cmap", "coolwarm_r")
        # TODO: based on residuals of last coordinate pair, but should use all residuals
        v = np.abs(np.nanpercentile(resid, [1, 99])).max()
        pcolormesh_kwargs.setdefault("vmin", -v)
        pcolormesh_kwargs.setdefault("vmax", v)

        return _plot_projections(
            grids=im_grids,
            ims=resid_ims,
            axes=axes,
            label=label,
            pcolormesh_kwargs=pcolormesh_kwargs,
        )


class StreamModel(ModelBase, abc.ABC):
    # Required: The name of the model component (e.g., "steam" or "background"):
    name = None

    # Required: A dictionary of coordinate names and corresponding Variable instances:
    variables = None

    # Required: A numpyro Distribution for the number of objects in this component:
    ln_N_dist = None

    # Optional: A dictionary of additional coordinate names to be passed to the
    # ln_prob() call when calling Variable.ln_prob(). For example, for a
    # Normal1DSplineVariable, the ln_prob() requires knowing the "x" values to evaluate
    # the spline at, so this might be {"pm1": {"x": "phi1", "y": "pm1"}}.
    data_required = None  # override optional

    # Optional: A dictionary of default grids to use when plotting each coordinate:
    default_grids = None

    def __init__(self, params):
        """
        Parameters
        ----------
        params : dict
            A nested dictionary of parameter values, which can either be values or a
            result of `numpyro.sample()` calls. The top-level keys should contain keys
            for `ln_N` and all `coord_names` (i.e. all keys of `.variables`). Parameters
            (values or dists) should be nested in sub-dictionaries keyed by parameter
            name.
        """

        # TODO: do this in a way that handles tuples
        # Validate input params:
        # for name in self.coord_names + ("ln_N",):
        #     if name not in params:
        #         raise ValueError(
        #             f"Expected coordinate name '{name}' in input parameters"
        #         )

        # store the inputted parameters
        self._pars = params

    def __init_subclass__(cls):
        if cls.name is None:
            msg = "You must set a name for this model component using the `name` class attribute."
            raise ValueError(msg)

        if cls.variables is None:
            msg = "You must define variables for this model by defining the dictionary `variables` to contain keys for each coordinate to model and values as instances of Component classes."
            raise ValueError(msg)

        # Now we validate the keys of .variables, and detect any joint distributions
        cls.coord_names = []
        cls._joint_names = {}
        for k in cls.variables:
            if isinstance(k, str):
                cls.coord_names.append(k)
                continue
            elif isinstance(k, tuple):
                # a shorthand name for the joint keys, joining the two variables
                cls._joint_names[k] = "__".join(k)

                for kk in k:
                    cls.coord_names.append(kk)

            else:
                msg = f"Invalid key type '{k}' in variables: type={type(k)}"
                raise ValueError(msg)
        cls._joint_names_inv = {v: k for k, v in cls._joint_names.items()}
        cls.coord_names = tuple(cls.coord_names)

        # TODO: Validate the data_required dictionary to make sure it has valid keys
        if cls.data_required is None:
            cls.data_required = {}

        # TODO: need to document assumption that data errors are passed in as _err keys
        cls._data_required = cls.data_required.copy()
        for k in cls.variables:
            if isinstance(k, str):
                cls._data_required = {
                    k: {"y": k, "y_err": f"{k}_err"} for k in cls.variables
                }
            else:  # assumed tuple/iterable
                err = tuple([f"{kk}_err" for kk in k])
                cls._data_required = {k: {"y": k, "y_err": err} for k in cls.variables}

        for k, v in cls.data_required.items():
            if k not in cls.variables:
                msg = f"Invalid data required key '{k}' (it doesn't exist in the variables dictionary)"
                raise ValueError(msg)
            cls._data_required[k] = v

        # Do this otherwise all subclasses will share the same mutables (i.e. dictionary
        # or strings) and modifying one will modify all:
        for name, thing in inspect.getmembers(cls):
            if inspect.isfunction(thing) or inspect.ismethod(thing):
                continue
            elif name.startswith("_") or name == "variables":
                continue
            setattr(cls, name, copy.deepcopy(getattr(cls, name)))

        # name value is required:
        if not cls.__name__.endswith("Base") and cls.name is None:
            msg = "you must specify a model component name"
            raise ValueError(msg)

        if cls.default_grids is None:
            cls.default_grids = {}

    ###################################################################################
    # Methods that can be overridden in subclasses:
    #
    def extra_ln_prior(self, params):
        """
        A log-prior to add to the total log-probability. This is useful for adding
        custom priors or regularizations that are not part of the model components.
        """
        return 0.0

    ###################################################################################
    # Shared methods for any density model component (or mixture):
    #
    @classmethod
    def setup_numpyro(cls, data=None):
        pars = {}

        # ln_N = ln(total number of stars in this component)
        pars["ln_N"] = numpyro.sample(f"{cls.name}-ln_N", cls.ln_N_dist)

        for comp_name, comp in cls.variables.items():
            name = cls._joint_names.get(comp_name, comp_name)
            pars[comp_name] = comp.setup_numpyro(name_prefix=f"{cls.name}-{name}-")

        obj = cls(params=pars)

        if data is not None:
            # Compute the log of the effective volume integral, used in the poisson
            # process likelihood
            ln_n = obj.ln_number_density(data)
            numpyro.factor(f"{cls.name}-factor-V", -obj.get_N())
            numpyro.factor(f"{cls.name}-factor-ln_n", ln_n.sum())
            numpyro.factor(f"{cls.name}-factor-extra_prior", obj.extra_ln_prior(pars))

        return obj

    def get_N(self):
        if self._pars is None:
            return None
        return jnp.exp(self._pars["ln_N"])

    def variable_ln_prob_density(self, data):
        """
        The log-probability for each component (e.g., phi2, pm1, etc) evaluated at the
        input data values.
        """
        # TODO: should be component_ln_prob_density to be consistent?
        comp_ln_probs = {}
        for comp_name, comp in self.variables.items():
            # NOTE: this silently ignores any data that is not present, even if expected
            # in the _data_required dict
            data_kw = {}
            for k, v in self._data_required[comp_name].items():
                if v in data:
                    data_kw[k] = data[v]
                elif isinstance(v, tuple):  # assumed to be a joint distribution
                    data_kw[k] = jnp.stack(jnp.array([data[vv] for vv in v])).T

            comp_ln_probs[comp_name] = comp.ln_prob(
                params=self._pars[comp_name], **data_kw
            )

        return comp_ln_probs

    def ln_prob_density(self, data):
        """
        The total log-probability evaluated at the input data values.
        """
        comp_ln_probs = self.variable_ln_prob_density(data)
        return jnp.sum(jnp.array(list(comp_ln_probs.values())), axis=0)

    def ln_number_density(self, data):
        return self._pars["ln_N"] + self.ln_prob_density(data)

    def ln_likelihood(self, data):
        """
        NOTE: This is the Poisson process likelihood
        """
        return -self.get_N() + self.ln_number_density(data).sum()

    @classmethod
    def _objective(cls, p, data, *args, **kwargs):
        """
        TODO: keys of inputs have to be normalized to remove tuples
        """
        p = cls._expand_variable_keys(p)
        data = cls._expand_variable_keys(data)
        model = cls(params=p, *args, **kwargs)
        ll = model.ln_likelihood(data) + model.extra_ln_prior(p)
        N = next(iter(data.values())).shape[0]
        return -ll / N

    ###################################################################################
    # Utilities for manipulating parameter names
    #
    @classmethod
    def _normalize_variable_keys(cls, input_dict):
        out = {}
        for k, v in input_dict.items():
            if isinstance(v, dict):
                v = cls._normalize_variable_keys(v)

            if k in cls._joint_names:
                out[cls._joint_names[k]] = v
            else:
                out[k] = v
        return out

    @classmethod
    def _expand_variable_keys(cls, input_dict):
        out = {}
        for k, v in input_dict.items():
            if isinstance(v, dict):
                v = cls._expand_variable_keys(v)

            if k in cls._joint_names_inv:
                out[cls._joint_names_inv[k]] = v
            else:
                out[k] = v
        return out

    @classmethod
    def _strip_model_name(cls, packed_pars):
        """
        Remove the model component name from the parameter names of a packed parameter
        dictionary.
        """
        return {k[len(cls.name) + 1 :]: v for k, v in packed_pars.items()}

    @classmethod
    def _unpack_params(cls, packed_pars):
        """
        Unpack a flat dictionary of parameters -- where keys have coordinate name,
        parameter name, and model component name -- into a nested dictionary with
        parameters grouped by coordinate name
        """
        packed_pars = cls._strip_model_name(packed_pars)

        pars = {}
        for k in packed_pars:
            if k == "ln_N":
                pars["ln_N"] = packed_pars["ln_N"]
                continue

            coord_name = k.split("_")[0]
            par_name = "_".join(k.split("_")[1:])
            if coord_name not in pars:
                pars[coord_name] = {}
            pars[coord_name][par_name] = packed_pars[k]

        return pars

    @classmethod
    def _pack_params(cls, pars):
        """
        Pack a nested dictionary of parameters into a flat dictionary where the keys
        correspond to the numpyro.sample() parameter names
        """
        packed_pars = {}
        packed_pars[f"{cls.name}-ln_N"] = pars["ln_N"]

        for k, v in pars.items():
            if k != "ln_N":
                for kk in v:
                    packed_pars[f"{cls.name}-{k}-{kk}"] = v[kk]

        return packed_pars

    ###################################################################################
    # Optimization
    #
    @classmethod
    def _get_jaxopt_bounds(cls):
        bounds_l = {}
        bounds_h = {}

        # ln_N special case
        bounds_l["ln_N"] = getattr(cls.ln_N_dist.support, "lower_bound", -jnp.inf)
        bounds_h["ln_N"] = getattr(cls.ln_N_dist.support, "upper_bound", jnp.inf)

        for k, comp in cls.variables.items():
            bounds = comp._param_bounds

            bounds_l[k] = {}
            bounds_h[k] = {}
            for par_name, sub_bounds in bounds.items():
                bounds_l[k][par_name] = sub_bounds[0]
                bounds_h[k][par_name] = sub_bounds[1]

        return (bounds_l, bounds_h)

    # OLD STUFF BELOW - SAVING FOR POSTERITY
    ###################################################################################
    # Optimization
    #
    # @classmethod
    # def optimize_numpyro(
    #     cls, data, init_params, seed=42, jaxopt_kwargs=None, use_bounds=True, **kwargs
    # ):
    #     """
    #     A wrapper around numpyro_ext.optim utilities, which enable jaxopt optimization
    #     for numpyro models.
    #     """
    #     from numpyro_ext.optim import optimize

    #     from .optim import CustomJAXOptBoundedMinimize, CustomJAXOptMinimize

    #     if jaxopt_kwargs is None:
    #         jaxopt_kwargs = {}
    #     jaxopt_kwargs.setdefault("maxiter", 2048)

    #     if use_bounds:
    #         jaxopt_kwargs.setdefault("method", "L-BFGS-B")
    #         bounds = cls._get_jaxopt_bounds()
    #         strategy = CustomJAXOptBoundedMinimize(
    #             loss_scale_factor=1 / len(data["phi1"]), bounds=bounds, **jaxopt_kwargs
    #         )
    #     else:
    #         jaxopt_kwargs.setdefault("method", "BFGS")
    #         strategy = CustomJAXOptMinimize(
    #             loss_scale_factor=1 / len(data["phi1"]), **jaxopt_kwargs
    #         )

    #     optimizer = optimize(
    #         cls.setup_numpyro,
    #         start=init_params,
    #         return_info=True,
    #         optimizer=strategy,
    #     )
    #     opt_pars, info = optimizer(jax.random.PRNGKey(seed), data=data, **kwargs)
    #     opt_pars = {k: v for k, v in opt_pars.items() if not k.startswith("obs_")}

    #     return cls.unpack_params(opt_pars, **kwargs), info


class StreamMixtureModel(ModelBase):
    def __init__(self, params, Components, tied_params=None):
        if tied_params is None:
            tied_params = []
        self.tied_params = list(tied_params)

        params = copy.deepcopy(params)
        for t1, t2 in self.tied_params:
            # t1 could be, e.g, ("stream", "pm1") or ("stream", "pm1", "mean")
            set_in_nested_dict(params, t1, get_from_nested_dict(params, t2))

        self.components = [C(params[C.name]) for C in Components]
        if len(self.components) < 1:
            msg = "You must pass at least one component"
            raise ValueError(msg)

        self.coord_names = None
        for component in self.components:
            if self.coord_names is None:
                self.coord_names = tuple(component.coord_names)
            else:
                if self.coord_names != tuple(component.coord_names):
                    msg = "All components must have the same set of coordinate names"
                    raise ValueError(msg)

        # TODO: same for default grids - should check that they are the same
        self.default_grids = component.default_grids

    @classmethod
    def setup_numpyro(cls, Components, data=None):
        components = []  # instances
        for Component in Components:
            components.append(Component.setup_numpyro(data=None))

        obj = cls(components)

        if data is not None:
            # Compute the log of the effective volume integral, used in the poisson
            # process likelihood
            ln_n = obj.ln_number_density(data)
            numpyro.factor(f"{cls.name}-factor-V", -obj.get_N())  # TODO: not defined
            numpyro.factor(f"{cls.name}-factor-ln_n", ln_n.sum())
            numpyro.factor(f"{cls.name}-factor-extra_prior", obj.extra_ln_prior())

        return obj

    def component_ln_prob_density(self, data):
        return {c.name: c.ln_prob_density(data) for c in self.components}

    def ln_prob_density(self, data):
        """
        The total log-probability evaluated at the input data values.
        """
        comp_ln_probs = self.component_ln_prob_density(data)
        return logsumexp(jnp.array(list(comp_ln_probs.values())), axis=0)

    def component_ln_number_density(self, data):
        return {c.name: c.ln_number_density(data) for c in self.components}

    def ln_number_density(self, data):
        comp_ln_n = self.component_ln_number_density(data)
        return logsumexp(jnp.array(list(comp_ln_n.values())), axis=0)

    def ln_likelihood(self, data):
        """
        NOTE: This is the Poisson process likelihood
        """
        # TODO: logsumexp instead?
        N = jnp.sum(jnp.array([c.get_N() for c in self.components]))
        return -N + self.ln_number_density(data).sum()

    def extra_ln_prior(self, params):
        return jnp.sum(
            jnp.array([c.extra_ln_prior(params[c.name]) for c in self.components])
        )

    def evaluate_on_2d_grids(self, grids=None, grid_coord_names=None):
        terms = {}
        for component in self.components:
            all_grids, component_terms = component.evaluate_on_2d_grids(
                grids=grids, grid_coord_names=grid_coord_names
            )
            for k, v in component_terms.items():
                if k not in terms:
                    terms[k] = []
                terms[k].append(v)

        terms = {k: logsumexp(jnp.array(v), axis=0) for k, v in terms.items()}
        return all_grids, terms

    @classmethod
    def _get_jaxopt_bounds(cls, Components, tied_params):
        bounds_l = {}
        bounds_h = {}
        for C in Components:
            _bounds = C._get_jaxopt_bounds()
            bounds_l[C.name] = C._normalize_variable_keys(_bounds[0])
            bounds_h[C.name] = C._normalize_variable_keys(_bounds[1])

        bounds = (bounds_l, bounds_h)

        if tied_params is not None:
            for b in bounds:
                for t1, _ in tied_params:
                    del_in_nested_dict(b, t1)

        return bounds

    @classmethod
    def _objective(cls, p, data, Components, tied_params=None):
        """
        TODO: keys of inputs have to be normalized to remove tuples
        """
        p = {C.name: C._expand_variable_keys(p[C.name]) for C in Components}

        data = Components[0]._expand_variable_keys(data)
        model = cls(params=p, Components=Components, tied_params=tied_params)
        ll = model.ln_likelihood(data) + model.extra_ln_prior(p)
        N = next(iter(data.values())).shape[0]
        return -ll / N

    @classmethod
    def optimize(
        cls,
        data,
        init_params,
        Components,
        jaxopt_kwargs=None,
        use_bounds=True,
        **kwargs,
    ):
        """
        A wrapper around numpyro_ext.optim utilities, which enable jaxopt optimization
        for numpyro models.
        """
        import jaxopt

        if jaxopt_kwargs is None:
            jaxopt_kwargs = {}
        jaxopt_kwargs.setdefault("maxiter", 8192)  # TODO: TOTALLY ARBITRARY

        tied_params = kwargs.pop("tied_params", None)

        optimize_kwargs = kwargs
        if use_bounds:
            jaxopt_kwargs.setdefault("method", "L-BFGS-B")
            optimize_kwargs["bounds"] = cls._get_jaxopt_bounds(Components, tied_params)
            Optimizer = jaxopt.ScipyBoundedMinimize
        else:
            jaxopt_kwargs.setdefault("method", "BFGS")
            Optimizer = jaxopt.ScipyMinimize

        optimizer = Optimizer(
            **jaxopt_kwargs,
            fun=partial(
                cls._objective,
                Components=Components,
                tied_params=tied_params,
            ),
        )
        opt_res = optimizer.run(
            init_params={
                C.name: C._normalize_variable_keys(init_params[C.name])
                for C in Components
            },
            data=Components[0]._normalize_variable_keys(data),
            **optimize_kwargs,
        )
        opt_p = {
            C.name: C._expand_variable_keys(opt_res.params[C.name]) for C in Components
        }
        return opt_p, opt_res.state
