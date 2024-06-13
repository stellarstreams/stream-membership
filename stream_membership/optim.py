"""
TODO: contribute to numpyro_ext?
"""

from numpyro.optim import _NumPyroOptim
from numpyro_ext.optim import _jaxopt_wrapper


class CustomJAXOptMinimize(_NumPyroOptim):
    """A NumPyro-compatible optimizer built using jaxopt.ScipyMinimize
    This exposes the ``ScipyMinimize`` optimizer from ``jaxopt`` to NumPyro. All
    keyword arguments are passed directly to ``jaxopt.ScipyMinimize``.
    """

    def __init__(self, loss_scale_factor=1.0, **kwargs):
        try:
            import jaxopt  # noqa
        except ImportError:
            msg = "jaxopt must be installed to use JAXOptMinimize"
            raise ImportError(msg)

        super().__init__(_jaxopt_wrapper)
        self.solver_kwargs = {} if kwargs is None else kwargs
        self._loss_scale = loss_scale_factor

    def eval_and_update(self, fn, in_state):
        from jaxopt import ScipyMinimize

        def loss(p):
            out, aux = fn(p)
            if aux is not None:
                msg = "JAXOptMinimize does not support models with mutable states."
                raise ValueError(msg)
            return out * self._loss_scale

        solver = ScipyMinimize(fun=loss, **self.solver_kwargs)
        out_state = solver.run(self.get_params(in_state))
        return (out_state.state.fun_val, None), (in_state[0] + 1, out_state)


class CustomJAXOptBoundedMinimize(_NumPyroOptim):
    """A NumPyro-compatible optimizer built using jaxopt.ScipyBoundedMinimize
    This exposes the ``ScipyBoundedMinimize`` optimizer from ``jaxopt`` to NumPyro. All
    keyword arguments are passed directly to ``jaxopt.ScipyBoundedMinimize``.
    """

    def __init__(self, loss_scale_factor=1.0, bounds=None, **kwargs):
        try:
            import jaxopt  # noqa
        except ImportError:
            msg = "jaxopt must be installed"
            raise ImportError(msg)

        super().__init__(_jaxopt_wrapper)
        self.solver_kwargs = {} if kwargs is None else kwargs
        self._loss_scale = loss_scale_factor
        self.bounds = bounds

    def eval_and_update(self, fn, in_state):
        from jaxopt import ScipyBoundedMinimize

        def loss(p):
            out, aux = fn(p)
            if aux is not None:
                msg = "JAXOptMinimize does not support models with mutable states."
                raise ValueError(msg)
            return out * self._loss_scale

        solver = ScipyBoundedMinimize(fun=loss, **self.solver_kwargs)
        out_state = solver.run(self.get_params(in_state), bounds=self.bounds)
        return (out_state.state.fun_val, None), (in_state[0] + 1, out_state)
