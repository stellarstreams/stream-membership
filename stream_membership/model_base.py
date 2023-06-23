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


class Normal1DSplineComponent:
    param_names = ("mean", "ln_std")

    def __init__(self, knots, bounds=None, param_bounds=None, spline_k=3):
        """
        Parameters:
        -----------
        knots : array-like
            Array of spline knot locations (i.e. the "x" locations).
        bounds : dict (optional)
            A dictionary with two optional keys: "low" or "high" to specify the lower
            and upper bounds of the component value (i.e. the "y" value bounds).
        param_bounds : dict (optional)
            A dictionary with keys set to any of the `param_names` to set prior bounds
            on the parameter values.
        spline_k : int (optional)
            The spline polynomial degree. Default is 3 (cubic splines).
        """
        if param_bounds is None:
            param_bounds = dict()

        self.bounds = tuple(bounds)

        self.spline_k = int(spline_k)
        self.knots = jnp.array(knots)

        # TODO: make this customizable?
        self._endpoints = "not-a-knot"

        # To be set when the model is initialized with set_params():
        self.splines = {}

    def set_params(self, params):
        for name in self.param_names:
            if name not in params:
                raise ValueError(
                    "You must pass in a value or numpyro dist for all parameters: "
                    f"{self.param_names}"
                )

        for name in self.param_names:
            self.splines[name] = InterpolatedUnivariateSpline(
                self.knots,
                params[name],
                k=self.spline_k,
                endpoints=self._endpoints,
            )

    def get_dist(self, eval_x):
        return dist.TruncatedNormal(
            loc=self.splines["mean"](eval_x),
            scale=jnp.exp(self.splines["ln_std"](eval_x)),
            low=self.bounds[0],
            high=self.bounds[1],
        )

    def ln_prob(self, x, y):
        d = self.get_dist(x)
        return d.log_prob(y)


class ModelBase(abc.ABC):
    # Name of the parameter that controls the log-total number of stars
    _num_name = "ln_N"

    # The name of the model component (e.g., "steam" or "background"):
    name = None  # required

    # TODO: A dictionary of ...
    components = None  # required

    def __init__(self, pars, **kwargs):
        """
        Base class.

        Parameters
        ----------
        pars : dict
            A nested dictionary of either (a) numpyro distributions, or (b) parameter
            values. The top-level keys should contain keys for `density_name` and all
            `coord_names`. Parameters (values or dists) should be nested in
            sub-dictionaries keyed by parameter name.
        """

        # Validate input params:
        for name in self.coord_names + (self._num_name,):
            if name not in pars:
                raise ValueError(
                    f"Expected coordinate name '{name}' in input parameters"
                )

        # store the input parameters, setup splines, and store data:
        self.pars = pars

        # TODO: instead, loop over components and set parameter values
        # self.splines = self.get_splines(self.pars)
        for name in self.coord_names:
            self.components[name].set_params(pars[name])

        self._setup_data(kwargs.get("data", None))

    @property
    def coord_names(self):
        return list(self.components.keys())

    def __init_subclass__(cls):
        if cls.name is None:
            raise ValueError(
                "You must set a name for this model component using the `name` class "
                "attribute."
            )

        if cls.components is None:
            raise ValueError(
                "You must define components for this model by defining the dictionary "
                "`components` to contain keys for each coordinate to model and values "
                "as instances of Component classes."
            )

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
    def _setup_data(self, data):
        # Note: data should be passed in / through by setup_numpyro(), but shouldn't be
        # passed as an argument when using the class otherwise:
        self._data = data

        # Validate input data:
        for coord_name in self.coord_names:
            if self._data is not None and coord_name not in self._data:
                raise ValueError(
                    f"Expected coordinate name '{coord_name}' in input data"
                )

        if self._data is not None:
            # Compute the log of the effective volume integral, used in the poisson
            # process likelihood
            ln_n = self.ln_number_density(self._data)
            numpyro.factor(f"V_{self.name}", -self.get_N())
            numpyro.factor(f"ln_n_{self.name}", ln_n.sum())
            numpyro.factor(f"extra_prior_{self.name}", self.extra_ln_prior())

    @classmethod
    def setup_numpyro(cls, data=None):
        pars = {}

        # ln_N = ln(total number of stars in this component)
        pars["ln_N"] = numpyro.sample(f"ln_N_{cls.name}", cls.ln_N_dist)

        # TODO: need to also support cases when comp_name is a tuple??
        for comp_name, comp in cls.components.items():
            pars[comp_name] = {}
            for par_name, prior in comp.priors.items():
                pars[comp_name][par_name] = numpyro.sample(
                    f"{cls.name}_{comp_name}_{par_name}",
                    prior,
                    sample_shape=prior.shape(),
                )

        return cls(pars=pars, data=data)

    def component_ln_prob_density(self, data):
        """
        The log-probability for each component (e.g., phi2, pm1, etc) evaluated at the
        input data values.
        """
        comp_ln_probs = {}
        for comp_name, comp in self.components:
            # TODO: method name?? log_prob vs. ln_prob in my style
            comp_ln_probs[comp_name] = comp.log_prob(data[comp_name])
        return comp_ln_probs

    def ln_prob_density(self, data):
        """
        The total log-probability evaluated at the input data values.
        """
        comp_ln_probs = self.component_ln_prob_density(data)
        return jnp.sum(jnp.array([v for v in comp_ln_probs.values()]), axis=0)

    def ln_number_density(self, data):
        # TODO: the thing I will want is to multiply p(phi1) by whatever component prob

        if return_terms:
            ln_probs = self.ln_prob_density(data, return_terms=True)
            ln_n0 = self.get_ln_n0(data)
            ln_n = {k: ln_n0 + ln_probs[k] for k in ln_probs}
        else:
            ln_prob = self.ln_prob_density(data, return_terms=False)
            ln_n = self.get_ln_n0(data) + ln_prob

        return ln_n

    def ln_likelihood(self, data):
        """
        NOTE: This is the Poisson process likelihood
        """
        ln_n = self.ln_prob_density(data)
        return -jnp.exp(self.get_ln_V()) + ln_n.sum()

    @classmethod
    def objective(cls, p, data):
        model = cls(p)
        ll = model.ln_likelihood(data)
        return -ll / len(data["phi1"])

    ###################################################################################
    # Evaluating on grids and plotting
    #
    def _get_grids_dict(self, grids, coord_names=None):
        if coord_names is None:
            coord_names = self.coord_names

        if grids is None:
            grids = {}
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

            ln_n = self.ln_number_density(tmp_data, return_terms=True)[name]
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
    def clip_params(cls, pars):
        """
        Clip the input parameter values so that they are within the requisite bounds
        """
        # TODO: tolerance MAGIC NUMBER 1e-2
        tol = 1e-2

        new_pars = {}
        new_pars[cls.density_name] = jnp.clip(
            pars[cls.density_name],
            cls.param_bounds[cls.density_name][0] + tol,
            cls.param_bounds[cls.density_name][1] - tol,
        )
        for coord_name in cls.coord_names:
            new_pars[coord_name] = {}
            for par_name in cls.param_bounds[coord_name]:
                new_pars[coord_name][par_name] = jnp.clip(
                    pars[coord_name][par_name],
                    cls.param_bounds[coord_name][par_name][0] + tol,
                    cls.param_bounds[coord_name][par_name][1] - tol,
                )
        return new_pars

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
            if k == cls.density_name:
                pars[cls.density_name] = packed_pars[cls.density_name]
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
    @classmethod
    def optimize(
        cls, data, init_params, seed=42, jaxopt_kwargs=None, use_bounds=True, **kwargs
    ):
        """
        A wrapper around numpyro_ext.optim utilities, which enable jaxopt optimization
        for numpyro models.
        """
        from numpyro_ext.optim import optimize

        from .optim import CustomJAXOptBoundedMinimize, CustomJAXOptMinimize

        if jaxopt_kwargs is None:
            jaxopt_kwargs = {}
        jaxopt_kwargs.setdefault("maxiter", 2048)

        if use_bounds:
            jaxopt_kwargs.setdefault("method", "L-BFGS-B")
            bounds = cls._get_jaxopt_bounds()
            strategy = CustomJAXOptBoundedMinimize(
                loss_scale_factor=1 / len(data["phi1"]), bounds=bounds, **jaxopt_kwargs
            )
        else:
            jaxopt_kwargs.setdefault("method", "BFGS")
            strategy = CustomJAXOptMinimize(
                loss_scale_factor=1 / len(data["phi1"]), **jaxopt_kwargs
            )

        optimizer = optimize(
            cls.setup_numpyro,
            start=init_params,
            return_info=True,
            optimizer=strategy,
        )
        opt_pars, info = optimizer(jax.random.PRNGKey(seed), data=data, **kwargs)
        opt_pars = {k: v for k, v in opt_pars.items() if not k.startswith("obs_")}

        return cls.unpack_params(opt_pars, **kwargs), info

    @classmethod
    def _get_jaxopt_bounds(cls):
        bounds_l = {}
        bounds_h = {}
        for k, bounds in cls.param_bounds.items():
            if k != "ln_n0" and k not in cls.coord_names:
                continue

            if isinstance(bounds, dict):
                bounds_l[k] = {}
                bounds_h[k] = {}
                for par_name, sub_bounds in bounds.items():
                    bounds_l[k][par_name] = jnp.full(
                        cls.shapes[k][par_name], sub_bounds[0]
                    )
                    bounds_h[k][par_name] = jnp.full(
                        cls.shapes[k][par_name], sub_bounds[1]
                    )

            else:
                bounds_l[k] = jnp.full(cls.shapes[k], bounds[0])
                bounds_h[k] = jnp.full(cls.shapes[k], bounds[1])

        return (bounds_l, bounds_h)


class StreamMixtureModel(ModelBase):
    name = "mixture"

    def __init__(self, components, **kwargs):
        self.coord_names = None

        self.components = list(components)
        if len(self.components) < 1:
            raise ValueError("You must pass at least one component")

        for component in self.components:
            if self.coord_names is None:
                self.coord_names = tuple(component.coord_names)
            else:
                if self.coord_names != tuple(component.coord_names):
                    raise ValueError("TODO")

        # TODO: same for default grids
        self.default_grids = component.default_grids

        self._setup_data(kwargs.get("data", None))

    @classmethod
    def setup_numpyro(cls, Components, data=None):
        components = []  # instances
        for Component in Components:
            components.append(Component.setup_numpyro(data=None))
        return cls(components, data=data)

    def get_ln_n0(self, data, return_total=True):
        ln_n0s = jnp.array([c.get_ln_n0(data) for c in self.components])
        if return_total:
            return logsumexp(ln_n0s, axis=0)
        else:
            return ln_n0s

    def get_ln_V(self, return_total=True):
        terms = jnp.array([c.get_ln_V() for c in self.components])
        if return_total:
            return logsumexp(terms, axis=0)
        else:
            return terms

    def get_dists(self, data):
        # TODO: this only works for 2D marginals!
        all_dists = [c.get_dists(data) for c in self.components]

        ln_n0s = self.get_ln_n0(data, return_total=False)
        total_ln_n0 = logsumexp(ln_n0s, axis=0)
        mix = dist.Categorical(
            probs=jnp.array([jnp.exp(ln_n0 - total_ln_n0) for ln_n0 in ln_n0s]).T
        )

        dists = {}
        for coord_name in self.coord_names:
            dists[coord_name] = dist.MixtureGeneral(
                mix,
                [tmp_dists[coord_name] for tmp_dists in all_dists],
            )

        return dists

    def ln_prob_density(self, data, return_terms=False):
        ln_n = self.ln_number_density(data, return_terms)
        total_ln_n0 = self.get_ln_n0(data, return_total=True)
        return ln_n - total_ln_n0

    def ln_number_density(self, data, return_terms=False):
        if return_terms:
            raise NotImplementedError("Sorry")

        ln_n0s = self.get_ln_n0(data, return_total=False)

        ln_ns = []
        for c, ln_n0 in zip(self.components, ln_n0s):
            ln_ns.append(ln_n0 + c.ln_prob_density(data, return_terms=False))

        return logsumexp(jnp.array(ln_ns), axis=0)

    @classmethod
    def objective(cls, p, Components, data):
        models = {C.name: C(p[C.name]) for C in Components}

        ln_ns = jnp.array([model.ln_number_density(data) for model in models.values()])
        ln_n = logsumexp(ln_ns, axis=0)

        V = jnp.sum(jnp.array([jnp.exp(model.get_ln_V()) for model in models.values()]))

        ll = -V + ln_n.sum()

        return -ll / len(data["phi1"])

    @classmethod
    def unpack_params(cls, pars, Components):
        pars_unpacked = {}
        for C in Components:
            pars_unpacked[C.name] = {}

        for par_name, par in pars.items():
            for C in Components:
                if par_name.endswith(C.name):
                    pars_unpacked[C.name][par_name] = par
                    break
        for C in Components:
            pars_unpacked[C.name] = C.unpack_params(pars_unpacked[C.name])
        return pars_unpacked

    def evaluate_on_grids(self, grids=None, coord_names=None):
        if coord_names is None:
            coord_names = self.coord_names
        if grids is None:
            grids = self.default_grids
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

            ln_ns = [
                c.ln_number_density(tmp_data, return_terms=True)[name]
                for c in self.components
            ]
            ln_n = logsumexp(jnp.array(ln_ns), axis=0)
            terms[name] = ln_n.reshape(grid1.shape)
            all_grids[name] = (grid1, grid2)

        return all_grids, terms

    def plot_knots(self, axes=None, **kwargs):
        if axes is None:
            import matplotlib.pyplot as plt

            _, axes = plt.subplots(
                len(self.coord_names) + 1,
                len(self.components),
                figsize=(6 * len(self.components), 3 * (len(self.coord_names) + 1)),
                sharex=True,
                constrained_layout=True,
            )

        for i, c in enumerate(self.components):
            c.plot_knots(axes=axes[:, i], **kwargs)

        return np.array(axes).flat[0].figure, axes

    @classmethod
    def _get_jaxopt_bounds(cls, Components):
        bounds_l = {}
        bounds_h = {}
        for Model in Components:
            _bounds = Model._get_jaxopt_bounds()
            bounds_l[Model.name] = _bounds[0]
            bounds_h[Model.name] = _bounds[1]
        bounds = (bounds_l, bounds_h)
        return bounds
