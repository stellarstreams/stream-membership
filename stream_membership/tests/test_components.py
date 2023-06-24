import jax
import numpy as np
import numpyro
import numpyro.distributions as dist
from numpyro.infer import Predictive

from ..components import Normal1DComponent, Normal1DSplineComponent


def test_normal1d():
    # Using numeric values so this can be evaluated:
    c = Normal1DComponent()
    c.set_params({"mean": 1.0, "ln_std": 0.5})
    c.ln_prob(5.0)
    c.ln_prob(np.linspace(0, 10, 15))

    # Using numpyro distributions for the parameters for deferred evaluation:
    def model():
        c = Normal1DComponent()
        c.set_params(
            {
                "mean": numpyro.sample("mean", dist.Uniform(0.0, 1.0)),
                "ln_std": numpyro.sample("ln_std", dist.Normal(0.5, 1.0)),
            }
        )

    predictive = Predictive(model, num_samples=10)
    samples = predictive(jax.random.PRNGKey(1))
    print(samples)
