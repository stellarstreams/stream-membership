import abc
import copy
import inspect

import jax
import jax.numpy as jnp
import numpy as np
import numpyro
import numpyro.distributions as dist
from jax.scipy.special import logsumexp

from .plot import _plot_projections


class ModelBase(abc.ABC):
    # The name of the model component (e.g., "steam" or "background"):
    name = None  # override required

    # A dictionary of coordinate names and corresponding model terms
    variables = None  # override required

    data_required = None  # override optional

    # TODO:
    ln_N_dist = None  # override required

    # TODO: optional??
    default_grids = None

    def __init__(self, pars):
        """
        Base class.

        Parameters
        ----------
        pars : dict
            A nested dictionary of either (a) numpyro distributions, or (b) parameter
            values. The top-level keys should contain keys for `ln_N` and all
            `coord_names`. Parameters (values or dists) should be nested in
            sub-dictionaries keyed by parameter name.
        """

        # Validate input params:
        for name in self.coord_names + ("ln_N",):
            if name not in pars:
                raise ValueError(
                    f"Expected coordinate name '{name}' in input parameters"
                )

        # store the inputted parameters
        self._pars = pars

    @property
    def coord_names(self):
        return tuple(self.variables.keys())

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
        cls._joint_names = {}
        for k in cls.variables:
            if isinstance(k, str):
                continue
            elif isinstance(k, tuple):
                # a shorthand name for the joint keys, joining the two variables
                cls._joint_names[k] = "_".join(k)
            else:
                raise ValueError(f"Invalid key type '{k}' in variables: type={type(k)}")

        # Validate the data_required dictionary to make sure ... TODO
        if cls.data_required is None:
            cls.data_required = {}

        cls._data_required = {k: {"y": k} for k in cls.variables}
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

    ###################################################################################
    # Methods that can be overridden in subclasses:
    #
    def extra_ln_prior(self):
        """
        TODO: describe why this would be useful
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
            numpyro.factor(f"{cls.name}-V", -obj.get_N())
            numpyro.factor(f"{cls.name}-ln_n", ln_n.sum())
            numpyro.factor(f"{cls.name}-extra_prior", obj.extra_ln_prior())

        return obj

    def get_N(self):
        if self._pars is None:
            return None
        return jnp.exp(self._pars["ln_N"])

    def component_ln_prob_density(self, data):
        """
        The log-probability for each component (e.g., phi2, pm1, etc) evaluated at the
        input data values.
        """
        comp_ln_probs = {}
        for comp_name, comp in self.variables.items():
            data_kw = {k: data[v] for k, v in self._data_required[comp_name].items()}
            comp_ln_probs[comp_name] = comp.ln_prob(
                params=self._pars[comp_name], **data_kw
            )
        return comp_ln_probs

    def ln_prob_density(self, data):
        """
        The total log-probability evaluated at the input data values.
        """
        comp_ln_probs = self.component_ln_prob_density(data)
        return jnp.sum(jnp.array([v for v in comp_ln_probs.values()]), axis=0)

    def component_ln_number_density(self, data):
        """
        NOTE: This computes the log-number density of the joint distribution n(phi1, X)
        where X is each component (e.g., phi2, pm1, etc.).
        """
        ln_probs = self.component_ln_prob_density(data)

        # TODO: this bakes in phi1 as a special name
        # TODO: this whole method needs a rethink! for the case of joints
        ln_prob_phi1 = ln_probs.pop("phi1")

        for k in ln_probs.keys():
            ln_probs[k] = self._pars["ln_N"] + ln_prob_phi1 + ln_probs[k]

        return ln_probs

    def ln_number_density(self, data):
        return self._pars["ln_N"] + self.ln_prob_density(data)

    def ln_likelihood(self, data):
        """
        NOTE: This is the Poisson process likelihood
        """
        return -self.get_N() + self.ln_number_density(data).sum()

    @classmethod
    def objective(cls, p, data):
        model = cls(p)
        ll = model.ln_likelihood(data)
        N = next(iter(data.values())).shape[0]
        return -ll / N

    ###################################################################################
    # Evaluating on grids and plotting
    #
    def _get_grids_dict(self, grids, coord_names=None):
        if coord_names is None:
            coord_names = self.coord_names

        if grids is None:
            grids = {}

        # TODO: again, phi1 is a special name!
        for name in ("phi1",) + coord_names:
            if name not in grids and name not in self.default_grids:
                raise ValueError(f"No default grid for {name}, so you must specify it")
            if name not in grids:
                grids[name] = self.default_grids.get(name)
        return grids

    def evaluate_on_grids(self, grids=None, coord_names=None):
        if coord_names is None:
            coord_names = self.coord_names
        grids = self._get_grids_dict(grids, coord_names)

        all_grids = {}
        terms = {}
        for name in coord_names:
            grid1, grid2 = np.meshgrid(grids["phi1"], grids[name])

            # Fill a data dict with zeros for all coordinates not being plotted
            # TODO: this is a hack and we take a performance hit for this because we
            # unnecessarily compute log-probs at nonsense values
            tmp_data = {"phi1": grid1.ravel()}
            for tmp_name in coord_names:
                if tmp_name == name:
                    tmp_data[tmp_name] = grid2.ravel()
                else:
                    tmp_data[tmp_name] = jnp.zeros_like(grid1.ravel())
                # TODO: hard-coded assumption that data errors are named _err
                tmp_data[f"{tmp_name}_err"] = jnp.zeros_like(grid1.ravel())

            ln_n = self.component_ln_number_density(tmp_data, return_terms=True)[name]
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
    ):
        if coord_names is None:
            coord_names = self.coord_names

        grids, ln_ns = self.evaluate_on_grids(grids=grids, coord_names=coord_names)
        ims = {name: np.exp(ln_ns[name]) for name in self.coord_names}
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
    ):
        from scipy.ndimage import gaussian_filter

        if coord_names is None:
            coord_names = self.coord_names

        grids = self._get_grids_dict(grids, coord_names)

        # Evaluate the model at the grid midpoints
        model_grids = {k: 0.5 * (g[:-1] + g[1:]) for k, g in grids.items()}
        im_grids, ln_ns = self.evaluate_on_grids(grids=model_grids)
        model_ims = {name: np.exp(ln_ns[name]) for name in self.coord_names}

        resid_ims = {}
        for name in self.coord_names:
            # get the number density: density=True is the prob density, so need to
            # multiply back in the total number of data points
            H_data, *_ = np.histogram2d(
                data["phi1"],
                data[name],
                bins=(grids["phi1"], grids[name]),
                density=True,
            )
            data_im = H_data.T * len(data["phi1"])

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

    ###################################################################################
    # Utilities for manipulating parameters
    #
    @classmethod
    def _normalize_variable_keys(cls, input_dict):
        out = {}
        for k, v in input_dict.items():
            if not isinstance(k, str):
                out["__".join(k)] = v
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
    def unpack_params(cls, packed_pars):
        """
        Unpack a flat dictionary of parameters -- where keys have coordinate name,
        parameter name, and model component name -- into a nested dictionary with
        parameters grouped by coordinate name
        """
        packed_pars = cls._strip_model_name(packed_pars)

        pars = {}
        for k in packed_pars.keys():
            if k == "ln_N":
                pars["ln_N"] = packed_pars["ln_N"]
                continue

            coord_name = k.split("_")[0]
            par_name = "_".join(k.split("_")[1:])
            if coord_name not in pars:
                pars[coord_name] = {}
            pars[coord_name][par_name] = packed_pars[k]

        return pars

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

    @classmethod
    def optimize(
        cls, data, init_params, seed=42, jaxopt_kwargs=None, use_bounds=True, **kwargs
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
            optimize_kwargs["bounds"] = cls._get_jaxopt_bounds()
            Optimizer = jaxopt.ScipyBoundedMinimize
        else:
            jaxopt_kwargs.setdefault("method", "BFGS")
            Optimizer = jaxopt.ScipyMinimize

        optimizer = Optimizer(**jaxopt_kwargs, fun=cls.objective)
        opt_res = optimizer.run(init_params=init_params, data=data, **optimize_kwargs)
        return opt_res.params, opt_res.state

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


# class MixtureModel(ModelBase):
#     name = "mixture"

#     def __init__(self, components, **kwargs):
#         self.coord_names = None

#         self.components = list(components)
#         if len(self.components) < 1:
#             raise ValueError("You must pass at least one component")

#         for component in self.components:
#             if self.coord_names is None:
#                 self.coord_names = tuple(component.coord_names)
#             else:
#                 if self.coord_names != tuple(component.coord_names):
#                     raise ValueError("TODO")

#         # TODO: same for default grids
#         self.default_grids = component.default_grids

#         self._setup_data(kwargs.get("data", None))

#     @classmethod
#     def setup_numpyro(cls, Components, data=None):
#         components = []  # instances
#         for Component in Components:
#             components.append(Component.setup_numpyro(data=None))
#         return cls(components, data=data)

#     def get_ln_n0(self, data, return_total=True):
#         ln_n0s = jnp.array([c.get_ln_n0(data) for c in self.components])
#         if return_total:
#             return logsumexp(ln_n0s, axis=0)
#         else:
#             return ln_n0s

#     def get_ln_V(self, return_total=True):
#         terms = jnp.array([c.get_ln_V() for c in self.components])
#         if return_total:
#             return logsumexp(terms, axis=0)
#         else:
#             return terms

#     def get_dists(self, data):
#         # TODO: this only works for 2D marginals!
#         all_dists = [c.get_dists(data) for c in self.components]

#         ln_n0s = self.get_ln_n0(data, return_total=False)
#         total_ln_n0 = logsumexp(ln_n0s, axis=0)
#         mix = dist.Categorical(
#             probs=jnp.array([jnp.exp(ln_n0 - total_ln_n0) for ln_n0 in ln_n0s]).T
#         )

#         dists = {}
#         for coord_name in self.coord_names:
#             dists[coord_name] = dist.MixtureGeneral(
#                 mix,
#                 [tmp_dists[coord_name] for tmp_dists in all_dists],
#             )

#         return dists

#     def ln_prob_density(self, data, return_terms=False):
#         ln_n = self.ln_number_density(data, return_terms)
#         total_ln_n0 = self.get_ln_n0(data, return_total=True)
#         return ln_n - total_ln_n0

#     def ln_number_density(self, data, return_terms=False):
#         if return_terms:
#             raise NotImplementedError("Sorry")

#         ln_n0s = self.get_ln_n0(data, return_total=False)

#         ln_ns = []
#         for c, ln_n0 in zip(self.components, ln_n0s):
#             ln_ns.append(ln_n0 + c.ln_prob_density(data, return_terms=False))

#         return logsumexp(jnp.array(ln_ns), axis=0)

#     @classmethod
#     def objective(cls, p, Components, data):
#         models = {C.name: C(p[C.name]) for C in Components}

#         ln_ns = jnp.array([model.ln_number_density(data) for model in models.values()])
#         ln_n = logsumexp(ln_ns, axis=0)

#         V = jnp.sum(jnp.array([jnp.exp(model.get_ln_V()) for model in models.values()]))

#         ll = -V + ln_n.sum()

#         return -ll / len(data["phi1"])

#     @classmethod
#     def unpack_params(cls, pars, Components):
#         pars_unpacked = {}
#         for C in Components:
#             pars_unpacked[C.name] = {}

#         for par_name, par in pars.items():
#             for C in Components:
#                 if par_name.endswith(C.name):
#                     pars_unpacked[C.name][par_name] = par
#                     break
#         for C in Components:
#             pars_unpacked[C.name] = C.unpack_params(pars_unpacked[C.name])
#         return pars_unpacked

#     def evaluate_on_grids(self, grids=None, coord_names=None):
#         if coord_names is None:
#             coord_names = self.coord_names
#         if grids is None:
#             grids = self.default_grids
#         grids = self._get_grids_dict(grids, coord_names)

#         all_grids = {}
#         terms = {}
#         for name in coord_names:
#             grid1, grid2 = np.meshgrid(grids["phi1"], grids[name])

#             # Fill a data dict with zeros for all coordinates not being plotted
#             # TODO: this is a hack and we take a performance hit for this because we
#             # unnecessarily compute log-probs at nonsense values
#             tmp_data = {"phi1": grid1.ravel()}
#             for tmp_name in coord_names:
#                 if tmp_name == name:
#                     tmp_data[tmp_name] = grid2.ravel()
#                 else:
#                     tmp_data[tmp_name] = jnp.zeros_like(grid1.ravel())
#                 # TODO: hard-coded assumption that data errors are named _err
#                 tmp_data[f"{tmp_name}_err"] = jnp.zeros_like(grid1.ravel())

#             ln_ns = [
#                 c.ln_number_density(tmp_data, return_terms=True)[name]
#                 for c in self.components
#             ]
#             ln_n = logsumexp(jnp.array(ln_ns), axis=0)
#             terms[name] = ln_n.reshape(grid1.shape)
#             all_grids[name] = (grid1, grid2)

#         return all_grids, terms

#     def plot_knots(self, axes=None, **kwargs):
#         if axes is None:
#             import matplotlib.pyplot as plt

#             _, axes = plt.subplots(
#                 len(self.coord_names) + 1,
#                 len(self.components),
#                 figsize=(6 * len(self.components), 3 * (len(self.coord_names) + 1)),
#                 sharex=True,
#                 constrained_layout=True,
#             )

#         for i, c in enumerate(self.components):
#             c.plot_knots(axes=axes[:, i], **kwargs)

#         return np.array(axes).flat[0].figure, axes

#     @classmethod
#     def _get_jaxopt_bounds(cls, Components):
#         bounds_l = {}
#         bounds_h = {}
#         for Model in Components:
#             _bounds = Model._get_jaxopt_bounds()
#             bounds_l[Model.name] = _bounds[0]
#             bounds_h[Model.name] = _bounds[1]
#         bounds = (bounds_l, bounds_h)
#         return bounds
