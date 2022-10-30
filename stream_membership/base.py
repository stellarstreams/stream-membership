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

__all__ = []


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

    ###################################################################################
    # Evaluating on grids and plotting
    #
    def _get_grids_dict(self, grids):
        if grids is None:
            grids = {}
        for name in ("phi1",) + self.coord_names:
            if name not in grids and name not in self.default_grids:
                raise ValueError(f"No default grid for {name}, so you must specify it")
            grids[name] = self.default_grids.get(name, grids.get(name))
        return grids

    def evaluate_on_grids(self, grids=None):
        grids = self._get_grids_dict(grids)

        all_grids = {}
        terms = {}
        for name in self.coord_names:
            grid1, grid2 = np.meshgrid(grids["phi1"], grids[name])

            # Fill a data dict with zeros for all coordinates not being plotted
            # TODO: this is a hack and we take a performance hit for this because we
            # unnecessarily compute log-probs at nonsense values
            tmp_data = {"phi1": grid1.ravel()}
            for tmp_name in self.coord_names:
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

    def _plot_projections(
        self,
        grids,
        ims,
        axes=None,
        label=True,
        pcolormesh_kwargs=None,
    ):
        import matplotlib as mpl

        _default_labels = {
            "phi2": r"$\phi_2$",
            "pm1": r"$\mu_{\phi_1}$",
            "pm2": r"$\mu_{\phi_2}$",
        }

        if pcolormesh_kwargs is None:
            pcolormesh_kwargs = {}

        if axes is None:
            import matplotlib.pyplot as plt

            _, axes = plt.subplots(
                len(self.coord_names),
                1,
                figsize=(10, 2 + 2 * len(self.coord_names)),
                sharex=True,
                sharey="row",
                constrained_layout=True,
            )

        if isinstance(axes, mpl.axes.Axes):
            axes = [axes]
        axes = np.array(axes)

        for i, name in enumerate(self.coord_names):
            grid1, grid2 = grids[name]
            axes[i].pcolormesh(
                grid1, grid2, ims[name], shading="auto", **pcolormesh_kwargs
            )
            axes[i].set_ylim(grid2.min(), grid2.max())

            if label:
                axes[i].set_ylabel(_default_labels[name])

        axes[0].set_xlim(grid1.min(), grid1.max())

        return axes.flat[0].figure, axes

    def plot_model_projections(
        self,
        grids=None,
        axes=None,
        label=True,
        pcolormesh_kwargs=None,
    ):
        grids, ln_ns = self.evaluate_on_grids(grids=grids)
        ims = {name: np.exp(ln_ns[name]) for name in self.coord_names}
        return self._plot_projections(
            grids=grids,
            ims=ims,
            axes=axes,
            label=label,
            pcolormesh_kwargs=pcolormesh_kwargs,
        )

    def plot_data_projections(
        self,
        data,
        grids=None,
        axes=None,
        label=True,
        smooth=1.0,
        pcolormesh_kwargs=None,
    ):
        from scipy.ndimage import gaussian_filter

        grids = self._get_grids_dict(grids)

        ims = {}
        im_grids = {}
        for name in self.coord_names:
            H_data, xe, ye = np.histogram2d(
                data["phi1"], data[name], bins=(grids["phi1"], grids[name])
            )
            if smooth is not None:
                H_data = gaussian_filter(H_data, smooth)
            im_grids[name] = (xe, ye)
            ims[name] = H_data.T

        return self._plot_projections(
            grids=im_grids,
            ims=ims,
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
    def optimize(cls, data, init_params, seed=42, **kwargs):
        """
        A wrapper around numpyro_ext.optim utilities, which enable jaxopt optimization
        for numpyro models.
        """
        from numpyro_ext.optim import optimize
        from .optim import CustomJAXOptMinimize

        strategy = CustomJAXOptMinimize(
            loss_scale_factor=1 / len(data['phi1']),
            method='BFGS',
        )
        optimizer = optimize(
            cls.setup_numpyro,
            start=init_params,
            return_info=True,
            optimizer=strategy
        )
        opt_pars, info = optimizer(jax.random.PRNGKey(seed), data=data, **kwargs)
        opt_pars = {k: v for k, v in opt_pars.items() if not k.startswith('obs_')}

        return cls.unpack_params(opt_pars, **kwargs), info


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

        # Note: data should be passed in / through by setup_numpyro(), but shouldn't be
        # passed as an argument when using the class otherwise:
        self.data = kwargs.get("data", None)

        # Validate input data:
        for coord_name in self.coord_names:
            if self.data is not None and coord_name not in self.data:
                raise ValueError(
                    f"Expected coordinate name '{coord_name}' in input data"
                )

        if self.data is not None:
            # Note: This is a required method that must be implemented in subclasses!
            dists = self.get_dists(self.data)

            # Compute the log of the effective volume integral, used in the poisson
            # process likelihood
            ln_V = self.get_ln_V()
            ln_n0 = self.splines["ln_n0"](self.data["phi1"])
            numpyro.factor(f"obs_ln_n0_{self.name}", -jnp.exp(ln_V) + ln_n0.sum())

            for coord_name in self.coord_names:
                numpyro.sample(
                    f"obs_{coord_name}_{self.name}",
                    dists[coord_name],
                    obs=self.data[coord_name],
                )

            numpyro.factor(f"extra_prior_{self.name}", self.extra_ln_prior())

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
            endpoints="natural",
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
                    endpoints="natural",
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
    def __init__(self, components, integration_grid_phi1=None, **kwargs):
        self.coord_names = None

        # TODO: validate that at least 1 compinent here
        self.components = components

        # Find the integration grid with the smallest step size:
        if integration_grid_phi1 is None:
            integration_grid_phi1 = components[0].integration_grid_phi1

        for component in self.components:
            if self.coord_names is None:
                self.coord_names = tuple(component.coord_names)
            else:
                if self.coord_names != tuple(component.coord_names):
                    raise ValueError("TODO")

        self.integration_grid_phi1 = integration_grid_phi1

        # TODO: same for default grids
        self.default_grids = component.default_grids

        # Note: data should be passed in / through by setup_numpyro(), but shouldn't be
        # passed as an argument when using the class otherwise:
        self.data = kwargs.get("data", None)

        if self.data is not None:
            dists = self.get_dists(self.data)

            ln_n0 = self.ln_number_density(self.data)
            factor = -jnp.exp(self.get_ln_V()) + ln_n0.sum()
            numpyro.factor("obs_ln_n0", factor)

            for coord_name in self.coord_names:
                numpyro.sample(
                    f"obs_{coord_name}", dists[coord_name], obs=self.data[coord_name]
                )

            for c in self.components:
                numpyro.factor(f"smooth_{c.name}", c.extra_ln_prior())

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
