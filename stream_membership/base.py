import abc
import copy
import inspect

import jax
import jax.numpy as jnp
import numpy as np
import numpyro
import numpyro.distributions as dist
from jax.scipy.special import logsumexp
from jax_cosmo.scipy.interpolate import InterpolatedUnivariateSpline

from .helpers import ln_simpson
from .plot import _plot_projections

__all__ = []


_endpoints = "not-a-knot"


class ModelBase(abc.ABC):

    ###################################################################################
    # Required methods that must be implemented in subclasses:
    #
    @abc.abstractclassmethod
    def setup_numpyro(cls):
        pass

    @abc.abstractmethod
    def get_dists(self):
        pass

    def extra_ln_prior(self):
        """
        This one is optional.
        """
        return 0.0

    ###################################################################################
    # Shared methods for any spline density model (component or mixture):
    #
    def _setup_data(self, data):
        # Note: data should be passed in / through by setup_numpyro(), but shouldn't be
        # passed as an argument when using the class otherwise:
        self.data = data

        # Validate input data:
        for coord_name in self.coord_names:
            if self.data is not None and coord_name not in self.data:
                raise ValueError(
                    f"Expected coordinate name '{coord_name}' in input data"
                )

        if self.data is not None:
            # Compute the log of the effective volume integral, used in the poisson
            # process likelihood
            ln_V = self.get_ln_V()
            ln_n = self.ln_number_density(self.data)
            numpyro.factor(f"V_{self.name}", -jnp.exp(ln_V))
            numpyro.factor(f"ln_n_{self.name}", ln_n.sum())
            numpyro.factor(f"extra_prior_{self.name}", self.extra_ln_prior())

    def get_ln_n0(self, data):
        return self.splines["ln_n0"](data["phi1"])

    def get_ln_V(self):
        return ln_simpson(
            self.splines["ln_n0"](self.integration_grid_phi1),
            x=self.integration_grid_phi1,
        )

    def ln_prob_density(self, data, return_terms=False):
        """
        TODO: only the prob. density of the likelihood
        """
        dists = self.get_dists(data)

        lls = {}
        for k in self.coord_names:
            lls[k] = dists[k].log_prob(data[k])

        if return_terms:
            return lls
        else:
            return jnp.sum(jnp.array([v for v in lls.values()]), axis=0)

    def ln_number_density(self, data, return_terms=False):
        """
        WARNING: When return_terms=True, you can't sum the coordinate dimensions
        """
        if return_terms:
            ln_probs = self.ln_prob_density(data, return_terms=True)
            ln_n0 = self.get_ln_n0(data)
            ln_n = {k: ln_n0 + ln_probs[k] for k in ln_probs}
        else:
            ln_prob = self.ln_prob_density(data, return_terms=False)
            ln_n = self.get_ln_n0(data) + ln_prob

        return ln_n

    def ln_likelihood(self, data):
        ln_n = self.ln_number_density(data, return_terms=False)
        return -jnp.exp(self.get_ln_V()) + ln_n.sum()

    @classmethod
    def objective(cls, p, data):
        model = cls(p)
        ll = model.ln_likelihood(data)
        return -ll / len(data['phi1'])

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
        coord_names=None
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
        coord_names=None
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
                data["phi1"], data[name], bins=(grids["phi1"], grids[name]), density=True
            )
            data_im = H_data.T * len(data['phi1'])

            resid_ims[name] = model_ims[name] - data_im

            if smooth is not None:
                resid_ims[name] = gaussian_filter(resid_ims[name], smooth)

        if pcolormesh_kwargs is None:
            pcolormesh_kwargs = {}
        pcolormesh_kwargs.setdefault('cmap', 'coolwarm_r')
        # TODO: hard-coded 10 - could be a percentile?
        pcolormesh_kwargs.setdefault('vmin', -10)
        pcolormesh_kwargs.setdefault('vmax', 10)

        return _plot_projections(
            grids=im_grids,
            ims=resid_ims,
            axes=axes,
            label=label,
            pcolormesh_kwargs=pcolormesh_kwargs,
        )

    def plot_knots(self, axes=None, add_legend=True):

        if axes is None:
            import matplotlib.pyplot as plt

            _, axes = plt.subplots(
                len(self.coord_names) + 1,
                1,
                figsize=(8, 4 * (len(self.coord_names) + 1)),
                sharex=True,
                constrained_layout=True,
            )

        spl = self.splines[self.density_name]
        (l,) = axes[0].plot(
            self.integration_grid_phi1,
            spl(self.integration_grid_phi1),
            marker="",
            label=self.density_name,
        )
        axes[0].scatter(
            self.knots[self.density_name],
            spl(self.knots[self.density_name]),
            color=l.get_color(),
        )
        axes[0].set_ylabel(self.density_name)

        for i, coord_name in enumerate(self.coord_names, start=1):
            ax = axes[i]

            for par_name, spl in self.splines.get(coord_name, {}).items():
                (l,) = ax.plot(
                    self.integration_grid_phi1,
                    spl(self.integration_grid_phi1),
                    label=f"{coord_name}: {par_name}",
                    marker="",
                )
                ax.scatter(
                    self.knots[coord_name],
                    spl(self.knots[coord_name]),
                    color=l.get_color(),
                )
            ax.set_ylabel(coord_name)

        if add_legend:
            for ax in axes:
                ax.legend(loc="best")

        axes[0].set_title(self.name)

        return axes[0].figure, axes

    ###################################################################################
    # Optimization
    #
    @classmethod
    def optimize(cls, data, init_params, seed=42, jaxopt_kwargs=None, use_bounds=True,
                 **kwargs):
        """
        A wrapper around numpyro_ext.optim utilities, which enable jaxopt optimization
        for numpyro models.
        """
        from numpyro_ext.optim import optimize
        from .optim import CustomJAXOptBoundedMinimize, CustomJAXOptMinimize

        if jaxopt_kwargs is None:
            jaxopt_kwargs = {}
        jaxopt_kwargs.setdefault('maxiter', 2048)

        if use_bounds:
            jaxopt_kwargs.setdefault('method', 'L-BFGS-B')
            bounds = cls._get_jaxopt_bounds()
            strategy = CustomJAXOptBoundedMinimize(
                loss_scale_factor=1 / len(data['phi1']),
                bounds=bounds,
                **jaxopt_kwargs
            )
        else:
            jaxopt_kwargs.setdefault('method', 'BFGS')
            strategy = CustomJAXOptMinimize(
                loss_scale_factor=1 / len(data['phi1']),
                **jaxopt_kwargs
            )

        optimizer = optimize(
            cls.setup_numpyro,
            start=init_params,
            return_info=True,
            optimizer=strategy,
        )
        opt_pars, info = optimizer(jax.random.PRNGKey(seed), data=data, **kwargs)
        opt_pars = {k: v for k, v in opt_pars.items() if not k.startswith('obs_')}

        return cls.unpack_params(opt_pars, **kwargs), info

    @classmethod
    def _get_jaxopt_bounds(cls):
        bounds_l = {}
        bounds_h = {}
        for k, bounds in cls.param_bounds.items():
            if k != 'ln_n0' and k not in cls.coord_names:
                continue

            if isinstance(bounds, dict):
                bounds_l[k] = {}
                bounds_h[k] = {}
                for par_name, sub_bounds in bounds.items():
                    bounds_l[k][par_name] = jnp.full(
                        cls.shapes[k][par_name],
                        sub_bounds[0]
                    )
                    bounds_h[k][par_name] = jnp.full(
                        cls.shapes[k][par_name],
                        sub_bounds[1]
                    )

            else:
                bounds_l[k] = jnp.full(
                    cls.shapes[k],
                    bounds[0]
                )
                bounds_h[k] = jnp.full(
                    cls.shapes[k],
                    bounds[1]
                )

        return (bounds_l, bounds_h)


class SplineDensityModelBase(ModelBase):
    # the name of the model component (e.g., "steam" or "background"):
    name = None  # required

    # containers to store bounds (limits) for coordinate values (e.g., phi1, phi2, etc.)
    # and bounds for parameters (e.g., for spline values ln_n0, mean phi2, etc.):
    coord_bounds = {}  # phi1 required, others are optional
    param_bounds = {}  # required for all parameters

    # the integration grid used to compute the effective volume integral used in the
    # poisson process likelihood:
    integration_grid_phi1 = None  # required

    # locations of spline knots and spline order for each parameter
    knots = {}  # required
    spline_ks = {}  # (optional)

    # the name of the parameters that control the linear density, and the names of the
    # coordinate components used in this model:
    density_name = "ln_n0"
    coord_names = ("phi2", "plx", "pm1", "pm2", "rv")

    def __init_subclass__(cls):
        # Do this otherwise all subclasses will share the same mutables (i.e. dictionary
        # or strings) and modifying one will modify all:
        for name, thing in inspect.getmembers(cls):
            if inspect.isfunction(thing) or inspect.ismethod(thing):
                continue
            elif name.startswith('_'):
                continue
            setattr(cls, name, copy.deepcopy(getattr(cls, name)))

        # name value is required:
        if not cls.__name__.endswith("Base") and cls.name is None:
            raise ValueError("you must specify a model component name")

        # validate the coordinate component bounds. phi1 bounds are required
        if "phi1" not in cls.coord_bounds:
            raise ValueError(
                "You must specify coordinate bounds for 'phi1' when defining "
                "a model component subclass by defining a class-level `coord_bounds` "
                "dictionary attribute."
            )
        cls.coord_names = tuple(cls.coord_names)
        for name in cls.coord_names:
            if name in cls.coord_bounds:
                assert cls.coord_bounds[name][1] > cls.coord_bounds[name][0]

        # Subclasses must define an integration grid in phi1 for computing the effective
        # volume in the poisson likelihood:
        if not cls.__name__.endswith("Base") and cls.integration_grid_phi1 is None:
            raise ValueError(
                "model component subclasses must specify an integration grid used to "
                "compute the effective volume term in the poisson process likelihood "
                "via the `integration_grid_phi1` class attribute"
            )
        else:
            # ensure it is a Jax array
            cls.integration_grid_phi1 = jnp.array(cls.integration_grid_phi1)

        # Fill out computed and default values for:
        # - shapes: expected parameter shapes
        # - spline_ks: spline order for each parameter
        cls.shapes = {}
        default_k = 3  # cubic splines by default

        cls.shapes[cls.density_name] = len(cls.knots[cls.density_name])
        cls.spline_ks[cls.density_name] = cls.spline_ks.get(cls.density_name, default_k)

        for coord_name in cls.coord_names:
            cls.shapes[coord_name] = {}
            tmp_ks = {}
            for par_name in cls.param_bounds[coord_name]:
                cls.shapes[coord_name][par_name] = len(cls.knots[coord_name])
                tmp_ks[par_name] = cls.spline_ks.get(coord_name, {}).get(
                    par_name, default_k
                )
            cls.spline_ks[coord_name] = tmp_ks

    def __init__(self, pars, **kwargs):
        """
        Base class for implementing density model components where parameters in the
        model are controlled by splines.

        Parameters
        ----------
        pars : dict
            A nested dictionary of either (a) numpyro distributions, or (b) parameter
            values. The top-level keys should contain keys for `density_name` and all
            `coord_names`. Parameters (values or dists) should be nested in
            sub-dictionaries keyed by parameter name.
        """

        # Validate input params:
        for name in self.coord_names + (self.density_name,):
            if name not in pars and name in self.knots:
                raise ValueError(
                    f"Expected coordinate name '{name}' in input parameters"
                )

        # Validate that input params have the right shapes:
        # TODO: doesn't work to be this strict - maybe need to just check last axis?
        # assert pars[self.density_name].shape == self.shapes[self.density_name]
        # for coord_name in self.coord_names:
        #     for par_name in self.shapes[coord_name]:
        #         assert (
        #             pars[coord_name][par_name].shape
        #             == self.shapes[coord_name][par_name]
        #         )

        # store the input parameters, setup splines, and store data:
        self.pars = pars
        self.splines = self.get_splines(self.pars)
        self._setup_data(kwargs.get("data", None))

    def get_splines(self, pars):
        """
        Set up splines for all parameters.

        This returns a nested dictionary of spline objects with `coord_names` as the
        top-level keys, and parameter names as keys of sub-dictionaries.
        """
        spls = {}

        spls[self.density_name] = InterpolatedUnivariateSpline(
            self.knots[self.density_name],
            pars[self.density_name],
            k=self.spline_ks[self.density_name],
            endpoints=_endpoints,
        )
        for coord_name in self.coord_names:
            if coord_name not in self.knots:
                continue

            spls[coord_name] = {}
            for par_name in pars[coord_name]:
                spls[coord_name][par_name] = InterpolatedUnivariateSpline(
                    self.knots[coord_name],
                    pars[coord_name][par_name],
                    k=self.spline_ks[coord_name][par_name],
                    endpoints=_endpoints,
                )

        return spls

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
        return {k[: -(len(cls.name) + 1)]: v for k, v in packed_pars.items()}

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


class SplineDensityMixtureModel(ModelBase):
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
        ln_n0s = jnp.array([c.splines["ln_n0"](data["phi1"]) for c in self.components])
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
            ln_ns.append(
                ln_n0 + c.ln_prob_density(data, return_terms=False)
            )

        return logsumexp(jnp.array(ln_ns), axis=0)

    @classmethod
    def objective(cls, p, Components, data):
        models = {C.name: C(p[C.name]) for C in Components}

        ln_ns = jnp.array([model.ln_number_density(data) for model in models.values()])
        ln_n = logsumexp(ln_ns, axis=0)

        V = jnp.sum(jnp.array([jnp.exp(model.get_ln_V()) for model in models.values()]))

        ll = -V + ln_n.sum()

        return -ll / len(data['phi1'])

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

