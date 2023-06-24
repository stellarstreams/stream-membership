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

    def __init__(self, coord_bounds=None):
        """
        Parameters:
        -----------
        coord_bounds : dict (optional)
            A dictionary with two optional keys: "low" or "high" to specify the lower
            and upper bounds of the component value (i.e. the "y" value bounds).
        """

        if coord_bounds is None:
            coord_bounds = (None, None)
        else:
            coord_bounds = tuple(coord_bounds)
        self.coord_bounds = coord_bounds

        self.params = None

    def set_params(self, pars, name_prefix=""):
        """ """
        self.params = {}
        for name in self.param_names:
            if name not in pars:
                raise ValueError(
                    "You must pass in a value or numpyro dist for all parameters: "
                    f"{self.param_names}"
                )

            if isinstance(pars[name], dist.Distribution):
                self.params[name] = numpyro.sample(
                    f"{name_prefix}{name}",
                    pars[name],
                    # sample_shape=pars[name].shape(),
                )
            else:
                self.params[name] = pars[name]

        return self.params

    def get_dist(self):
        raise NotImplementedError()

    def ln_prob(self, y, *args, **kwargs):
        d = self.get_dist(*args, **kwargs)
        return d.log_prob(y)


class Normal1DComponent(ComponentBase):
    param_names = ("mean", "ln_std")

    def get_dist(self):
        return dist.TruncatedNormal(
            loc=self.params["mean"],
            scale=jnp.exp(self.params["ln_std"]),
            low=self.coord_bounds[0],
            high=self.coord_bounds[1],
        )


class Normal1DSplineComponent(ComponentBase):
    param_names = ("mean", "ln_std")

    def __init__(self, knots, spline_k=3, coord_bounds=None):
        """
        Parameters:
        -----------
        knots : array-like
            Array of spline knot locations (i.e. the "x" locations).
        spline_k : int (optional)
            The spline polynomial degree. Default is 3 (cubic splines).
        coord_bounds : dict (optional)
            A dictionary with two optional keys: "low" or "high" to specify the lower
            and upper bounds of the component value (i.e. the "y" value bounds).
        """

        self.spline_k = int(spline_k)
        self.knots = jnp.array(knots)

        # TODO: make this customizable?
        self._endpoints = "not-a-knot"

        # To be set when the model is initialized with set_params():
        self.splines = {}

        super().__init__(coord_bounds=coord_bounds)

    def set_params(self, params):
        pars = super().set_params(params)

        for name in self.param_names:
            self.splines[name] = InterpolatedUnivariateSpline(
                self.knots,
                pars[name],
                k=self.spline_k,
                endpoints=self._endpoints,
            )

    def get_dist(self, x):
        return dist.TruncatedNormal(
            loc=self.splines["mean"](x),
            scale=jnp.exp(self.splines["ln_std"](x)),
            low=self.coord_bounds[0],
            high=self.coord_bounds[1],
        )


class GridGMMComponent(ComponentBase):
    param_names = ("ws",)

    def __init__(self, locs, scales, coord_bounds=None):
        """
        Parameters:
        -----------
        locs : array-like
        scales : array-like (optional)
        coord_bounds : dict (optional)
            A dictionary with two optional keys: "low" or "high" to specify the lower
            and upper bounds of the component value (i.e. the "y" value bounds).
        """
        self.locs = jnp.array(locs)
        self.scales = jnp.array(scales)
        for name in ["locs", "scales"]:
            if getattr(self, name).ndim != 2:
                raise ValueError("locs and scales must be 2D arrays.")

        super().__init__(coord_bounds=coord_bounds)

    def get_dist(self):
        return TruncatedGridGMM(
            mixing_distribution=dist.Categorical(self.params["ws"]),
            locs=self.locs,
            scales=self.scales,
            low=self.coord_bounds[0],
            high=self.coord_bounds[1],
        )
