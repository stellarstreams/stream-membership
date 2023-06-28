import abc
import copy
import inspect
from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
import numpyro
import numpyro.distributions as dist
from jax.scipy.special import logsumexp

from .plot import _plot_projections


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

        optimize_kwargs = {}
        if use_bounds:
            jaxopt_kwargs.setdefault("method", "L-BFGS-B")
            optimize_kwargs["bounds"] = cls._get_jaxopt_bounds()
            optimize_kwargs["bounds"] = (
                cls._normalize_variable_keys(optimize_kwargs["bounds"][0]),
                cls._normalize_variable_keys(optimize_kwargs["bounds"][1]),
            )
            Optimizer = jaxopt.ScipyBoundedMinimize
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
    def _get_grids_dict(self, grids, coord_names=None):
        if coord_names is None:
            coord_names = self.coord_names

        if grids is None:
            grids = {}

        for name in coord_names:
            if name not in grids and name not in self.default_grids:
                raise ValueError(f"No default grid for {name}, so you must specify it")
            if name not in grids:
                grids[name] = self.default_grids.get(name)
        return grids

    def evaluate_on_2d_grids(self, grids=None, x_coord="phi1", coord_names=None):
        """
        Evaluate the log-number density on a 2D grid of coordinates. This is useful for
        creating plots of the predicted model number density.
        """
        if coord_names is None:
            coord_names = self.coord_names
        grids = self._get_grids_dict(grids, coord_names)

        all_grids = {}
        terms = {}
        for name in coord_names:
            if name == x_coord:
                continue

            grid1, grid2 = np.meshgrid(grids[x_coord], grids[name])

            bin_area = np.abs(
                (grids[x_coord][1] - grids[x_coord][0])
                * (grids[name][1] - grids[name][0])
            )

            # Fill a data dict with zeros for all coordinates not being plotted
            # TODO: this is a hack and we take a performance hit for this because we
            # unnecessarily compute log-probs at nonsense values
            tmp_data = {x_coord: grid1.ravel()}
            for tmp_name in coord_names:
                if tmp_name == name:
                    tmp_data[tmp_name] = grid2.ravel()
                elif tmp_name == x_coord:
                    tmp_data[tmp_name] = grid1.ravel()
                else:
                    tmp_data[tmp_name] = jnp.zeros_like(grid1.ravel())

                # TODO: hard-coded assumption that data errors are named _err
                # tmp_data[f"{tmp_name}_err"] = jnp.zeros_like(grid1.ravel())

            ln_ps = self.variable_ln_prob_density(tmp_data)
            ln_n = self._pars["ln_N"] + ln_ps[x_coord] + ln_ps[name] + np.log(bin_area)
            terms[name] = ln_n.reshape(grid1.shape)
            all_grids[name] = (grid1, grid2)

        return all_grids, terms

    def plot_model_projections(
        self,
        grids=None,
        axes=None,
        label=True,
        pcolormesh_kwargs=None,
        coord_names=None,
        x_coord="phi1",
    ):
        if coord_names is None:
            coord_names = self.coord_names

        grids, ln_ns = self.evaluate_on_2d_grids(
            grids=grids, coord_names=coord_names, x_coord=x_coord
        )
        ims = {
            name: np.exp(ln_ns[name]) for name in self.coord_names if name != x_coord
        }
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
        axes=None,
        label=True,
        smooth=1.0,
        pcolormesh_kwargs=None,
        coord_names=None,
        x_coord="phi1",
    ):
        from scipy.ndimage import gaussian_filter

        if coord_names is None:
            coord_names = self.coord_names

        grids = self._get_grids_dict(grids, coord_names)

        # Evaluate the model at the grid midpoints
        model_grids = {k: 0.5 * (g[:-1] + g[1:]) for k, g in grids.items()}
        im_grids, ln_ns = self.evaluate_on_2d_grids(grids=model_grids, x_coord=x_coord)
        model_ims = {
            name: np.exp(ln_ns[name]) for name in self.coord_names if name != x_coord
        }

        resid_ims = {}
        for name in self.coord_names:
            if name == x_coord:
                continue

            # get the number density: density=True is the prob density, so need to
            # multiply back in the total number of data points
            # TODO: slightly wrong because histogram2d takes edges, but want to evaluate
            # on grid centers
            H_data, *_ = np.histogram2d(
                data["phi1"],
                data[name],
                bins=(grids["phi1"], grids[name]),
            )
            data_im = H_data.T

            resid_ims[name] = model_ims[name] - data_im

            if smooth is not None:
                resid_ims[name] = gaussian_filter(resid_ims[name], smooth)

        if pcolormesh_kwargs is None:
            pcolormesh_kwargs = {}
        pcolormesh_kwargs.setdefault("cmap", "coolwarm_r")
        # TODO: hard-coded 10 - could be a percentile?
        pcolormesh_kwargs.setdefault("vmin", -10)
        pcolormesh_kwargs.setdefault("vmax", 10)

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
            raise ValueError(
                "You must set a name for this model component using the `name` class "
                "attribute."
            )

        if cls.variables is None:
            raise ValueError(
                "You must define variables for this model by defining the dictionary "
                "`variables` to contain keys for each coordinate to model and values "
                "as instances of Component classes."
            )

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
                raise ValueError(f"Invalid key type '{k}' in variables: type={type(k)}")
        cls._joint_names_inv = {v: k for k, v in cls._joint_names.items()}
        cls.coord_names = tuple(cls.coord_names)

        # Validate the data_required dictionary to make sure it's valid TODO
        if cls.data_required is None:
            cls.data_required = {}

        # TODO: need to document assumption that data errors are passed in as _err keys
        cls._data_required = {k: {"y": k, "y_err": f"{k}_err"} for k in cls.variables}
        for k, v in cls.data_required.items():
            if k not in cls.variables:
                raise ValueError(
                    f"Invalid data required key '{k}' (it doesn't exist in the "
                    "variables dictionary)"
                )
            cls._data_required[k] = v

        # Do this otherwise all subclasses will share the same mutables (i.e. dictionary
        # or strings) and modifying one will modify all:
        for name, thing in inspect.getmembers(cls):
            if inspect.isfunction(thing) or inspect.ismethod(thing):
                continue
            elif name.startswith("_"):
                continue
            setattr(cls, name, copy.deepcopy(getattr(cls, name)))

        # name value is required:
        if not cls.__name__.endswith("Base") and cls.name is None:
            raise ValueError("you must specify a model component name")

        if cls.default_grids is None:
            cls.default_grids = {}

    ###################################################################################
    # Methods that can be overridden in subclasses:
    #
    def extra_ln_prior(self):
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

        obj = cls(pars=pars)

        if data is not None:
            # Compute the log of the effective volume integral, used in the poisson
            # process likelihood
            ln_n = obj.ln_number_density(data)
            numpyro.factor(f"{cls.name}-factor-V", -obj.get_N())
            numpyro.factor(f"{cls.name}-factor-ln_n", ln_n.sum())
            numpyro.factor(f"{cls.name}-factor-extra_prior", obj.extra_ln_prior())

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
        comp_ln_probs = {}
        for comp_name, comp in self.variables.items():
            # NOTE: this silently ignores any data that is not present, even if expected
            # in the _data_required dict
            data_kw = {
                k: data[v]
                for k, v in self._data_required[comp_name].items()
                if v in data
            }
            comp_ln_probs[comp_name] = comp.ln_prob(
                params=self._pars[comp_name], **data_kw
            )
        return comp_ln_probs

    def ln_prob_density(self, data):
        """
        The total log-probability evaluated at the input data values.
        """
        comp_ln_probs = self.variable_ln_prob_density(data)
        return jnp.sum(jnp.array([v for v in comp_ln_probs.values()]), axis=0)

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
        ll = model.ln_likelihood(data)
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
    # Utilities for manipulating parameters
    #
    # Below only needed for optimize_numpyro()
    # @classmethod
    # def _strip_model_name(cls, packed_pars):
    #     """
    #     Remove the model component name from the parameter names of a packed parameter
    #     dictionary.
    #     """
    #     return {k[len(cls.name) + 1 :]: v for k, v in packed_pars.items()}

    # @classmethod
    # def unpack_params(cls, packed_pars):
    #     """
    #     Unpack a flat dictionary of parameters -- where keys have coordinate name,
    #     parameter name, and model component name -- into a nested dictionary with
    #     parameters grouped by coordinate name
    #     """
    #     packed_pars = cls._strip_model_name(packed_pars)

    #     pars = {}
    #     for k in packed_pars.keys():
    #         if k == "ln_N":
    #             pars["ln_N"] = packed_pars["ln_N"]
    #             continue

    #         coord_name = k.split("_")[0]
    #         par_name = "_".join(k.split("_")[1:])
    #         if coord_name not in pars:
    #             pars[coord_name] = {}
    #         pars[coord_name][par_name] = packed_pars[k]

    #     return pars

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
    def __init__(self, params, Components):
        self.components = [C(params[C.name]) for C in Components]
        if len(self.components) < 1:
            raise ValueError("You must pass at least one component")

        self.coord_names = None
        for component in self.components:
            if self.coord_names is None:
                self.coord_names = tuple(component.coord_names)
            else:
                if self.coord_names != tuple(component.coord_names):
                    raise ValueError(
                        "All components must have the same set of coordinate names"
                    )

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
            numpyro.factor(f"{cls.name}-factor-V", -obj.get_N())
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
        return logsumexp(
            jnp.array([ln_prob for ln_prob in comp_ln_probs.values()]), axis=0
        )

    def component_ln_number_density(self, data):
        return {c.name: c.ln_number_density(data) for c in self.components}

    def ln_number_density(self, data):
        comp_ln_n = self.component_ln_number_density(data)
        return logsumexp(jnp.array([ln_n for ln_n in comp_ln_n.values()]), axis=0)

    def ln_likelihood(self, data):
        """
        NOTE: This is the Poisson process likelihood
        """
        # TODO: logsumexp instead?
        N = jnp.sum(jnp.array([c.get_N() for c in self.components]))
        return -N + self.ln_number_density(data).sum()

    def evaluate_on_2d_grids(self, grids=None, x_coord="phi1", coord_names=None):
        terms = {}
        for component in self.components:
            all_grids, component_terms = component.evaluate_on_2d_grids(
                grids, x_coord, coord_names
            )
            for k, v in component_terms.items():
                if k not in terms:
                    terms[k] = []
                terms[k].append(v)

        terms = {k: logsumexp(jnp.array(v), axis=0) for k, v in terms.items()}
        return all_grids, terms

    @classmethod
    def _get_jaxopt_bounds(cls, Components):
        bounds_l = {}
        bounds_h = {}
        for C in Components:
            _bounds = C._get_jaxopt_bounds()
            bounds_l[C.name] = C._normalize_variable_keys(_bounds[0])
            bounds_h[C.name] = C._normalize_variable_keys(_bounds[1])
        bounds = (bounds_l, bounds_h)
        return bounds

    @classmethod
    @partial(jax.jit, static_argnums=(0, 3))
    def _objective(cls, p, data, Components):
        """
        TODO: keys of inputs have to be normalized to remove tuples

        TODO: just realized a problem with this: if one component has a joint for phi1,
        phi2 and others don't, this won't work
        """
        p = {C.name: C._expand_variable_keys(p[C.name]) for C in Components}

        # TODO see note about above one component having a joint for phi1, phi2
        data = Components[0]._expand_variable_keys(data)
        model = cls(params=p, Components=Components)
        ll = model.ln_likelihood(data)
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
        jaxopt_kwargs.setdefault("maxiter", 1024)  # TODO: TOTALLY ARBITRARY

        optimize_kwargs = {}
        if use_bounds:
            jaxopt_kwargs.setdefault("method", "L-BFGS-B")
            optimize_kwargs["bounds"] = cls._get_jaxopt_bounds(Components)
            Optimizer = jaxopt.ScipyBoundedMinimize
        else:
            jaxopt_kwargs.setdefault("method", "BFGS")
            Optimizer = jaxopt.ScipyMinimize

        optimizer = Optimizer(
            **jaxopt_kwargs,
            fun=partial(cls._objective, Components=Components),
        )
        opt_res = optimizer.run(
            init_params={
                C.name: C._normalize_variable_keys(init_params[C.name])
                for C in Components
            },
            data=Components[0]._normalize_variable_keys(
                data
            ),  # TODO: see issue about joint phi1, phi2
            **optimize_kwargs,
        )
        opt_p = {
            C.name: C._expand_variable_keys(opt_res.params[C.name]) for C in Components
        }
        return opt_p, opt_res.state
