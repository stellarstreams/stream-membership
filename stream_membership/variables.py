from functools import partial

import jax
import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist
from jax_cosmo.scipy.interpolate import InterpolatedUnivariateSpline

from .truncatedgridgmm import TruncatedGridGMM

__all__ = [
    "Normal1DVariable",
    "Normal1DSplineVariable",
    "GridGMMVariable",
    "UniformVariable",
]


class VariableBase:
    param_names = None

    def __init_subclass__(cls) -> None:
        required_methods = ["get_dist"]
        for f in required_methods:
            if getattr(cls, f) is getattr(__class__, f):
                raise ValueError(
                    "Subclasses of VariableBase must implement methods for: "
                    f"{required_methods}"
                )

        if cls.param_names is None:
            raise ValueError(
                "Subclasses of VariableBase must specify an iterable of string "
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
            lb = jnp.broadcast_to(lb, self.param_priors[name].batch_shape)
            ub = jnp.broadcast_to(ub, self.param_priors[name].batch_shape)
            self._param_bounds[name] = (lb, ub)

    def setup_numpyro(self, name_prefix=""):
        """ """
        pars = {}

        # TODO: what if in both?
        for name, prior in self.param_priors.items():
            pars[name] = numpyro.sample(f"{name_prefix}{name}", prior)

        return pars

    def get_dist(self, params, y_err=0.0):
        raise NotImplementedError()

    @partial(jax.jit, static_argnums=(0,))
    def ln_prob(self, params, y, *args, **kwargs):
        d = self.get_dist(params, *args, **kwargs)
        return d.log_prob(y.reshape((-1,) + d.event_shape))


class UniformVariable(VariableBase):
    param_names = ()

    def get_dist(self, params, y_err=0.0):
        return dist.Uniform(*self.coord_bounds)


class Normal1DVariable(VariableBase):
    param_names = ("mean", "ln_std")

    def get_dist(self, params, y_err=0.0):
        return dist.TruncatedNormal(
            loc=params["mean"],
            scale=jnp.sqrt(jnp.exp(2 * params["ln_std"]) + y_err**2),
            low=self.coord_bounds[0],
            high=self.coord_bounds[1],
        )


class Normal1DSplineVariable(VariableBase):
    param_names = ("mean", "ln_std")

    def __init__(self, param_priors, knots, spline_ks=3, coord_bounds=None):
        """
        Parameters:
        -----------
        knots : array-like
            Array of spline knot locations (i.e. the "x" locations).
        spline_ks : int, dict (optional)
            The spline polynomial degree. Default is 3 (cubic splines). Pass in a dict
            with keys matching the parameter names to control the degree of each spline
            separately.
        coord_bounds : tuple (optional)
            An iterable with two elements to specify the lower and upper bounds of the
            component value s (i.e. the "y" value bounds).
        """

        if isinstance(spline_ks, int):
            self.spline_ks = {k: spline_ks for k in self.param_names}
        elif isinstance(spline_ks, dict):
            self.spline_ks = {k: spline_ks.get(k, 3) for k in self.param_names}
        else:
            raise TypeError("Invalid type for spline_ks - must be int or dict")
        self.knots = jnp.array(knots)

        # TODO: make this customizable?
        self._endpoints = "not-a-knot"

        super().__init__(param_priors=param_priors, coord_bounds=coord_bounds)

    def get_dist(self, params, x, y_err=0.0):
        self.splines = {}
        for name in self.param_names:
            self.splines[name] = InterpolatedUnivariateSpline(
                self.knots,
                params[name],
                k=self.spline_ks[name],
                endpoints=self._endpoints,
            )

        return dist.TruncatedNormal(
            loc=self.splines["mean"](x),
            scale=jnp.sqrt(jnp.exp(2 * self.splines["ln_std"](x)) + y_err**2),
            low=self.coord_bounds[0],
            high=self.coord_bounds[1],
        )


class Normal1DSplineMixtureVariable(Normal1DSplineVariable):
    param_names = ("w", "mean1", "ln_std1", "mean2", "ln_std2")

    def get_dist(self, params, x, y_err=0.0):
        self.splines = {}
        for name in self.param_names:
            self.splines[name] = InterpolatedUnivariateSpline(
                self.knots,
                params[name],
                k=self.spline_ks[name],
                endpoints=self._endpoints,
            )

        locs = jnp.stack(
            jnp.array([self.splines["mean1"](x), self.splines["mean2"](x)])
        ).T

        var1 = jnp.exp(2 * self.splines["ln_std1"](x))
        var2 = var1 + jnp.exp(2 * self.splines["ln_std2"](x))
        scales = jnp.sqrt(jnp.stack(jnp.array([var1 + y_err**2, var2 + y_err**2]))).T

        w = self.splines["w"](x)
        probs = jnp.stack(jnp.array([w, 1 - w]))

        return dist.MixtureSameFamily(
            dist.Categorical(probs=probs.T),
            dist.TruncatedNormal(
                loc=locs,
                scale=scales,
                low=self.coord_bounds[0],
                high=self.coord_bounds[1],
            ),
        )


class GridGMMVariable(VariableBase):
    param_names = ("zs",)

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

        self._stick = dist.transforms.StickBreakingTransform()

    def get_dist(self, params, y_err=None):
        if y_err is None:
            scales = self.scales
        else:
            scales = jnp.sqrt(self.scales[None] ** 2 + y_err[:, None] ** 2)
        return TruncatedGridGMM(
            mixing_distribution=dist.Categorical(probs=self._stick(params["zs"])),
            locs=self.locs,
            scales=scales,
            low=self.coord_bounds[0],
            high=self.coord_bounds[1],
        )
