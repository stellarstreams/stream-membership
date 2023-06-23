import jax.numpy as jnp
import numpyro.distributions as dist
from jax_cosmo.scipy.interpolate import InterpolatedUnivariateSpline

__all__ = ["Normal1DSplineComponent"]


class ComponentBase:
    param_names = None

    def __init_subclass__(cls) -> None:
        required_methods = ["set_params", "get_dist", "ln_prob"]
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

    def set_params(self):
        raise NotImplementedError()

    def get_dist(self):
        raise NotImplementedError()

    def ln_prob(self):
        raise NotImplementedError()


class Normal1DSplineComponent(ComponentBase):
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
