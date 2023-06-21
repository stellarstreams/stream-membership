import copy
import inspect

import jax.numpy as jnp
from jax_cosmo.scipy.interpolate import InterpolatedUnivariateSpline

from .helpers import ln_simpson
from .model_base import ModelBase

_endpoints = "not-a-knot"


class SplineDensityModelBase(ModelBase):
    # Parameters to be set by subclasses:

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
            elif name.startswith("_"):
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

    def get_ln_n0(self, data):
        return self.splines["ln_n0"](data["phi1"])

    def get_ln_V(self):
        return ln_simpson(
            self.splines["ln_n0"](self.integration_grid_phi1),
            x=self.integration_grid_phi1,
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
