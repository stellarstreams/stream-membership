import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist
from jax_cosmo.scipy.interpolate import InterpolatedUnivariateSpline

from .truncatedgridgmm import TruncatedGridGMM

__all__ = ["Normal1DComponent", "Normal1DSplineComponent", "GridGMMComponent"]


class ComponentBase:
    param_names = None

    def __init_subclass__(cls) -> None:
        required_methods = ["get_dist"]
        for f in required_methods:
            if getattr(cls, f) is getattr(__class__, f):
                raise ValueError(
                    "Subclasses of ComponentBase must implement methods for: "
                    f"{required_methods}"
                )

        if cls.param_names is None:
            raise ValueError(
                "Subclasses of ComponentBase must specify an iterable of string "
                "parameter names as the `param_names` attribute."
            )
        cls.param_names = tuple(cls.param_names)

    def __init__(self, param_priors, coord_bounds=None):
        """
        Parameters:
        -----------
        coord_bounds : tuple (optional)
            An iterable with two elements to specify the lower and upper bounds of the
            component value s (i.e. the "y" value bounds).
        """

        if coord_bounds is None:
            coord_bounds = (None, None)
        else:
            coord_bounds = tuple(coord_bounds)
        self.coord_bounds = coord_bounds

        if param_priors is None:
            param_priors = {}
        self.param_priors = dict(param_priors)

        # to be filled below with parameter bounds
        self._param_bounds = dict()

        # check that all expected param names are specified:
        for name in self.param_names:
            if name not in self.param_priors:
                raise ValueError(
                    f"Missing parameter: {name} - you must specify a prior for all "
                    "parameters"
                )

            # bounds based on support of prior
            lb = getattr(self.param_priors[name].support, "lower_bound", -jnp.inf)
            ub = getattr(self.param_priors[name].support, "upper_bound", jnp.inf)
            self._param_bounds[name] = (lb, ub)

    def setup_numpyro(self, name_prefix=""):
        """ """
        pars = {}

        # TODO: what if in both?
        for name, prior in self.param_priors.items():
            pars[name] = numpyro.sample(f"{name_prefix}{name}", prior)

        return pars

    def get_dist(self, params):
        raise NotImplementedError()

    def ln_prob(self, params, y, *args, **kwargs):
        d = self.get_dist(params, *args, **kwargs)
        return d.log_prob(y)


class Normal1DComponent(ComponentBase):
    param_names = ("mean", "ln_std")

    def get_dist(self, params, *_, **__):
        return dist.TruncatedNormal(
            loc=params["mean"],
            scale=jnp.exp(params["ln_std"]),
            low=self.coord_bounds[0],
            high=self.coord_bounds[1],
        )


class Normal1DSplineComponent(ComponentBase):
    param_names = ("mean", "ln_std")

    def __init__(self, param_priors, knots, spline_k=3, coord_bounds=None):
        """
        Parameters:
        -----------
        knots : array-like
            Array of spline knot locations (i.e. the "x" locations).
        spline_k : int (optional)
            The spline polynomial degree. Default is 3 (cubic splines).
        coord_bounds : tuple (optional)
            An iterable with two elements to specify the lower and upper bounds of the
            component value s (i.e. the "y" value bounds).
        """

        self.spline_k = int(spline_k)
        self.knots = jnp.array(knots)

        # TODO: make this customizable?
        self._endpoints = "not-a-knot"

        super().__init__(param_priors=param_priors, coord_bounds=coord_bounds)

    def get_dist(self, params, x, *_, **__):
        # TODO: I think this has to go here, and not in init
        self.splines = {}
        for name in self.param_names:
            self.splines[name] = InterpolatedUnivariateSpline(
                self.knots,
                params[name],
                k=self.spline_k,
                endpoints=self._endpoints,
            )

        return dist.TruncatedNormal(
            loc=self.splines["mean"](x),
            scale=jnp.exp(self.splines["ln_std"](x)),
            low=self.coord_bounds[0],
            high=self.coord_bounds[1],
        )


class GridGMMComponent(ComponentBase):
    param_names = ("ws",)

    def __init__(self, param_priors, locs, scales, coord_bounds=None):
        """
        Parameters:
        -----------
        locs : array-like
        scales : array-like (optional)
        coord_bounds : tuple (optional)
            An iterable with two elements to specify the lower and upper bounds of the
            component value s (i.e. the "y" value bounds).
        """
        self.locs = jnp.array(locs)
        self.scales = jnp.array(scales)
        for name in ["locs", "scales"]:
            if getattr(self, name).ndim != 2:
                raise ValueError(f"{name} must be a 2D array.")

        super().__init__(param_priors=param_priors, coord_bounds=coord_bounds)

    def get_dist(self, params, x, *_, **__):
        return TruncatedGridGMM(
            mixing_distribution=dist.Categorical(params["ws"]),
            locs=self.locs,
            scales=self.scales,
            low=self.coord_bounds[0],
            high=self.coord_bounds[1],
        )
