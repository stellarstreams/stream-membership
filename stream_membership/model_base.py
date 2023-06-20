import abc

import jax
import jax.numpy as jnp
import numpy as np
import numpyro
import numpyro.distributions as dist
from jax.scipy.special import logsumexp

from .plot import _plot_projections


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
    # Shared methods for any density model component (or mixture):
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
