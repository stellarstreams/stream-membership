__all__ = ["ModelMixin", "ModelComponent", "ComponentMixtureModel"]

import copy
from abc import abstractmethod
from collections import defaultdict
from itertools import chain
from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
import matplotlib.axes as mpl_axes
import numpy as np
import numpyro
import numpyro.distributions as dist
from jax.typing import ArrayLike
from jax_ext.integrate import ln_simpson
from numpyro.handlers import seed

from stream_membership.distributions import ConcatenatedDistributions

from ._typing import CoordinateName
from .plot import _plot_projections
from .utils import get_coord_from_data_dict


class ModelMixin:
    """
    Generic functionality for model component and mixture model, like evaluating on
    grids and plotting
    """

    def _get_grids_2d(
        self,
        grids_1d: dict[str, ArrayLike],
        grid_coord_names: list[tuple[str, str]],
    ) -> dict[tuple[str, str], list[jax.Array]]:
        """
        Takes a dictionary of 1D grids and returns a dictionary of 2D grids for each
        pair of coordinates in grid_coord_names, which will be used to evaluate and plot
        the model in 2D projections.
        """
        grids_2d = {}
        for name_pair in grid_coord_names:
            for name in name_pair:
                if name not in grids_1d:
                    msg = (
                        f"You must specify a 1D grid for the component '{name}' via "
                        "the grids_1d argument"
                    )
                    raise ValueError(msg)
            grids_2d[name_pair] = jnp.meshgrid(*[grids_1d[name] for name in name_pair])

        return grids_2d

    @abstractmethod
    def evaluate_on_2d_grids(
        self, pars, grids, grid_coord_names, x_coord_name
    ) -> tuple[
        dict[tuple[str, str], list[jax.Array]], dict[tuple[str, str], jax.Array]
    ]:
        pass

    def plot_model_projections(
        self,
        pars: dict[str, Any],
        grids: dict[str, ArrayLike],
        ndata: dict[str, Any] = None, #for scaling to data when making data+model+residual plots
        grid_coord_names: list[tuple[str, str]] | None = None,
        x_coord_name: str | None = None,
        axes: mpl_axes.Axes | None = None,
        label: bool = True,
        pcolormesh_kwargs: dict | None = None,
    ):
        """
        Plot the model evaluated on 2D grids.

        Parameters
        ----------
        data
            A dictionary of data arrays, where the keys are the names of the coordinates
            in the model component.
        pars
            A dictionary of parameter values for the model component.
        grids
            A dictionary of 1D grids for each coordinate in the model component. The
            keys should be the names of the coordinates you want to evaluate the model
            on, and must always contain the x coordinate.
        grid_coord_names
            A list of tuples of coordinate names to evaluate the model on. The default
            is to pair the x coordinate with each other coordinate in the model
            component. For example, if the model component has coordinates "phi1",
            "phi2", and "pm1", the default grid_coord_names would be [("phi1", "phi2"),
            ("phi1", "pm1")].
        x_coord_name
            The name of the x coordinate to use for evaluating the model. If None, the
            default x coordinate will be used, which is taken to be the 0th coordinate
            name in the specified "coord_distributions".
        axes
            A matplotlib axes object to plot the residuals on. If None, a new figure and
            axes will be created.
        label
            Whether to add labels to the axes.
        pcolormesh_kwargs
            Keyword arguments to pass to the matplotlib.pcolormesh() function.
        """
        grids, ln_ps = self.evaluate_on_2d_grids(
            pars=pars,
            grids=grids,
            grid_coord_names=grid_coord_names,
            x_coord_name=x_coord_name,
        )

        if ndata == None:
            ims = {k: np.exp(v) for k, v in ln_ps.items()}

        else:
            # Compute the bin area for each 2D grid cell for a cheap integral...
            bin_area = {
                k: np.abs(np.diff(grid1[0])[None] * np.diff(grid2[:, 0])[:, None])
                for k, (grid1, grid2) in grids.items()
            }

            ln_ns = {
                k: ln_p + np.log(ndata) + np.log(bin_area[k]) for k, ln_p in ln_ps.items()
            }
            ims = {k: np.exp(v) for k, v in ln_ns.items()}

        return _plot_projections(
            grids=grids,
            ims=ims,
            axes=axes,
            label=label,
            pcolormesh_kwargs=pcolormesh_kwargs,
        )

    def plot_residual_projections(
        self,
        data: dict[str, Any],
        pars: dict[str, Any],
        grids: dict[str, ArrayLike],
        grid_coord_names: list[tuple[str, str]] | None = None,
        x_coord_name: str | None = None,
        axes: mpl_axes.Axes | None = None,
        label: bool = True,
        pcolormesh_kwargs: dict | None = None,
        smooth: int | float | None = 1.0,
    ):
        """
        Plot the residuals of the model evaluated on 2D grids compared to the input
        data, binned into the same 2D grids.

        Parameters
        ----------
        data
            A dictionary of data arrays, where the keys are the names of the coordinates
            in the model component.
        pars
            A dictionary of parameter values for the model component.
        grids
            A dictionary of 1D grids for each coordinate in the model component. The
            keys should be the names of the coordinates you want to evaluate the model
            on, and must always contain the x coordinate.
        grid_coord_names
            A list of tuples of coordinate names to evaluate the model on. The default
            is to pair the x coordinate with each other coordinate in the model
            component. For example, if the model component has coordinates "phi1",
            "phi2", and "pm1", the default grid_coord_names would be [("phi1", "phi2"),
            ("phi1", "pm1")].
        x_coord_name
            The name of the x coordinate to use for evaluating the model. If None, the
            default x coordinate will be used, which is taken to be the 0th coordinate
            name in the specified "coord_distributions".
        axes
            A matplotlib axes object to plot the residuals on. If None, a new figure and
            axes will be created.
        label
            Whether to add labels to the axes.
        pcolormesh_kwargs
            Keyword arguments to pass to the matplotlib.pcolormesh() function.
        smooth
            The standard deviation of the Gaussian kernel to use for smoothing the
            residuals. If None, no smoothing is applied.

        """
        from scipy.ndimage import gaussian_filter

        grids_2d, ln_ps = self.evaluate_on_2d_grids(
            pars=pars,
            grids=grids,
            grid_coord_names=grid_coord_names,
            x_coord_name=x_coord_name,
        )
        N_data = next(iter(data.values())).shape[0]

        # Compute the bin area for each 2D grid cell for a cheap integral...
        bin_area = {
            k: np.abs(np.diff(grid1[0])[None] * np.diff(grid2[:, 0])[:, None])
            for k, (grid1, grid2) in grids_2d.items()
        }

        ln_ns = {
            k: ln_p + np.log(N_data) + np.log(bin_area[k]) for k, ln_p in ln_ps.items()
        }
        model_ims = {k: np.exp(v) for k, v in ln_ns.items()}

        resid_ims = {}
        for name_pair, model_im in model_ims.items():
            # get the number density: density=True is the prob density, so need to
            # multiply back in the total number of data points
            H_data, *_ = np.histogram2d(
                data[name_pair[0]],
                data[name_pair[1]],
                bins=(grids[name_pair[0]], grids[name_pair[1]]),
            )
            data_im = H_data.T

            resid = model_im - data_im
            resid_ims[name_pair] = resid

            if smooth is not None:
                resid_ims[name_pair] = gaussian_filter(resid_ims[name_pair], smooth)

        if pcolormesh_kwargs is None:
            pcolormesh_kwargs = {}
        pcolormesh_kwargs.setdefault("cmap", "coolwarm_r")
        # TODO: based on residuals of last coordinate pair, but should use all residuals
        v = np.abs(np.nanpercentile(resid, [1, 99])).max()
        pcolormesh_kwargs.setdefault("vmin", -v)
        pcolormesh_kwargs.setdefault("vmax", v)

        return _plot_projections(
            grids=grids_2d,
            ims=resid_ims,
            axes=axes,
            label=label,
            pcolormesh_kwargs=pcolormesh_kwargs,
        )


@jax.tree_util.register_pytree_node_class
class CoordinateMapping(dict):
    """Note: This is needed because we use a mix of tuples and strings as keys in the
    field dictionaries in ModelComponent below.
    See: https://github.com/jax-ml/jax/issues/15358

    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def tree_flatten(self):
        sorted_keys = sorted(
            self.keys(), key=lambda k: k[0] if isinstance(k, tuple) else k
        )
        return (tuple(self[k] for k in sorted_keys), sorted_keys)

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        tmp = cls()
        for k, v in zip(aux_data, children, strict=True):
            tmp[k] = v
        return tmp


class ModelComponent(eqx.Module, ModelMixin):
    """
    Creating evaluating the different components of the density model.

    :param name:
        the name of the model component (usually either 'background', 'stream', or 'offtrack')
    :type name: str

    :param coord_distributions:
        a dictionary of the distributions of the component parameters.
        The keys are the names of the component parameters
        (e.g. `'phi1'`, `'phi2'`, `'mu_phi1'`, `'mu_phi2'`, `('phi1', 'phi2')`, etc.)
    :type coord_distributions: dict[str | tuple, Any]

    :param coord_parameters:
        a dictionary of the parameters of the distributions in `coord_distributions`.
        The keys are the names of the component parameters (the keys in `coord_distributions`)
        and the values are dictionaries containing the parameters for the distributions.
        For example, a truncated normal distribution (`dist.TruncatedNormal` in numpyro)
        might have the parameters loc, scale, low, and high.
    :type coord_parameters: dict[str | tuple, dict[str, dist.Distribution | tuple | ArrayLike | dict]]

    :param default_x_coord:
        (optional) the default x-coordinate for the model component. default=None
    :type default_x_coord: str | None

    :param conditional_data:
        (optional) a dictionary of any additional data that is required for evaluating the
        log-probability of a coordinate's probability distribution. For example, a
        spline-enabled distribution might require the phi1 data to evaluate the spline
        at the phi1 values.
        The keys are the names of the component parameters
        (e.g. `'phi1'`, `'phi2'`, `'mu_phi1'`, `'mu_phi2'`, `('phi1', 'phi2')`, etc.)
        and the values are dictionaries of the conditional data for those parameters. default=None
    :type conditional_data: dict[CoordinateName, dict[str, str]]

    :attr:`_coord_names`
        the names of the component parameters (the keys in `coord_distributions` and `coord_parameters`)
    :type _coord_names: list[str]

    :attr:`_sample_order`
        the order in which the component parameters should be sampled
    :type _sample_order: list[CoordinateName]
    """

    name: str
    coord_distributions: dict[CoordinateName, Any]
    coord_parameters: dict[
        CoordinateName, dict[str, dist.Distribution | tuple | ArrayLike | dict]
    ]
    default_x_coord: str | None = None
    conditional_data: dict[CoordinateName, dict[str, str]] = eqx.field(default=None)
    _coord_names: list[str] = eqx.field(init=False)
    _sample_order: list[CoordinateName] = eqx.field(init=False)

    def __init__(
        self,
        name,
        coord_distributions,
        coord_parameters,
        default_x_coord=None,
        conditional_data=None,
    ):
        self.name = name
        self.coord_distributions = CoordinateMapping(coord_distributions)
        self.coord_parameters = CoordinateMapping(coord_parameters)
        self.default_x_coord = default_x_coord
        self.conditional_data = conditional_data

        # def __post_init__(self):
        # Validate that the keys (i.e. coordinate names) in coord_distributions and
        # coord_parameters are the same
        if set(self.coord_distributions.keys()) != set(self.coord_parameters.keys()):
            msg = "Keys in coord_distributions and coord_parameters must match"
            raise ValueError(msg)

        if self.default_x_coord is None:
            self.default_x_coord = next(iter(self.coord_distributions.keys()))
            if not isinstance(self.default_x_coord, str):
                self.default_x_coord = self.default_x_coord[0]

        self._coord_names = []
        for coord_name in self.coord_distributions:
            if isinstance(coord_name, tuple):
                self._coord_names.extend(coord_name)
            else:
                self._coord_names.append(coord_name)

        # This is used to specify any extra data that is required for evaluating the
        # log-probability of a coordinate's probability distribution. For example, a
        # spline-enabled distribution might require the phi1 data to evaluate the spline
        # at the phi1 values
        if self.conditional_data is None:
            self.conditional_data = {}

        # Validate that there are no circular dependencies:
        _pairs = []
        for coord_name in self.coord_distributions:
            for val in self.conditional_data.get(coord_name, {}).values():
                _pairs.append((coord_name, val))
        for _pair in _pairs:
            if _pair[::-1] in _pairs:
                msg = f"Circular dependency: {_pair}"
                raise ValueError(msg)
        self._sample_order = self._make_sample_order()

        # Validate that coordinate names can't have "-" and component, coordinate, and
        # parameter names can't have ":"
        if any("-" in name for name in self._coord_names):
            msg = "Coordinate names can't contain '-'"
            raise ValueError(msg)

        if any(":" in name for name in self._coord_names) or ":" in self.name:
            msg = "Coordinate names and component names can't contain ':'"
            raise ValueError(msg)

    @property
    def coord_names(self):
        return self._coord_names

    def _make_numpyro_name(
        self, coord_name: CoordinateName, arg_name: str | None = None
    ) -> str:
        """
        Convert a nested set of component name (this class name), coordinate name, and
        parameter name into a single string for naming a parameter with
        numpyro.sample().

        Parameters
        ----------
        coord_name
            The name of the coordinate in the component. If a coordinate can only be
            modeled as a joint, pass a tuple of strings.
        arg_name
            The name of the parameter used in the model component for the coordinate.
        """
        if isinstance(coord_name, tuple):
            coord_name = "-".join(coord_name)

        name = f"{self.name}:{coord_name}"
        if arg_name is None:
            return name
        return f"{name}:{arg_name}"

    def _expand_numpyro_name(self, numpyro_name: str) -> tuple[str, str | tuple, str]:
        """
        Convert a numpyro name into a tuple of component name, coordinate name, and
        parameter name.

        Parameters
        ----------
        numpyro_name
            The name of the parameter in the numpyro model, i.e. the name of a parameter
            specified with numpyro.sample(). In the context of this model, this should
            be something like "background:phi2:loc", where the model is named
            "background", the coordinate is named "phi2", and the parameter is named
            "loc".
        """
        bits = numpyro_name.split(":")
        return (
            bits[0],
            tuple(bits[1].split("-")) if "-" in bits[1] else bits[1],
            bits[2],
        )

    def pack_params(self, pars: dict[CoordinateName, Any]) -> dict[str, Any]:
        """
        Convert a dictionary of parameters as a nested dictionary into a flat dictionary
        with packed numpyro-compatible names.

        Parameters
        ----------
        pars
            A dictionary of parameters where the keys are the names of the coordinates
            in the model component and the values are dictionaries of the parameters for
            each coordinate.
        """
        packed_pars: dict[str, Any] = {}

        for coord_name, sub_pars in pars.items():
            for arg_name, val in sub_pars.items():
                numpyro_name = self._make_numpyro_name(coord_name, arg_name)
                packed_pars[numpyro_name] = val

        return packed_pars

    def expand_numpyro_params(
        self, pars: dict[str, Any], skip_invalid: bool = False
    ) -> dict[str | tuple, Any]:
        """
        Convert a dictionary of numpyro parameters into a nested dictionary where the
        keys are the coordinate names and parameter name.

        Parameters
        ----------
        pars
            A dictionary of numpyro parameters where the keys are the names of the
            parameters created with numpyro.sample().
        """
        expanded_pars: dict[str, dict] = {}
        for k, v in pars.items():
            try:
                name, coord_name, arg_name = self._expand_numpyro_name(k)
            except Exception as e:
                if not skip_invalid:
                    raise e
                continue

            if name not in expanded_pars:
                expanded_pars[name] = {}
            if coord_name not in expanded_pars[name]:
                expanded_pars[name][coord_name] = {}
            expanded_pars[name][coord_name][arg_name] = v

        return expanded_pars[name]

    def make_dists(
        self,
        pars: dict[CoordinateName, Any] | None = None,
        dists: dict[CoordinateName, dist.Distribution] | None = None,
        ) -> dict[str | tuple, Any]:

        """
        Make a dictionary of distributions for each coordinate in the component.

        Parameters
        ----------
        pars
            A dictionary of parameters to pass to the numpyro.sample() calls that
            create the distributions. The dictionary should be structured as follows:
            {
                "coord_name": {
                    "arg_name": value
                }
            }
            where "coord_name" is the name of the coordinate and "arg_name" is the name
            of the argument to pass to the numpyro.sample() call that creates the
            distribution. "value" is the value to pass to the numpyro.sample() call.
        """

        pars = pars if pars is not None else {}
        dists = dists if dists is not None else {}

        for coord_name, Distribution in self.coord_distributions.items():
            kwargs = {}

            if coord_name in dists:
                continue

            for arg, val in self.coord_parameters.get(coord_name, {}).items():
                numpyro_name = self._make_numpyro_name(coord_name, arg)

                # Note: passing in a tuple as a value is a way to wrap the value in a
                # function or outer distribution, for example for a mixture model
                if isinstance(val, tuple) and callable(val[0]):
                    wrapper, val = val  # noqa: PLW2901
                else:
                    wrapper = lambda x: x  # noqa: E731

                if arg in pars.get(coord_name, {}):
                    # If an argument is passed in the pars dictionary, use that value.
                    # This is useful, for example, for constructing the coordinate
                    # distributions once a model is optimized or sampled, so you can
                    # pass in parameter values to evaluate the model.
                    par = pars[coord_name][arg]
                elif isinstance(val, dict):
                    par = numpyro.sample(numpyro_name, **val)
                elif isinstance(val, dist.Distribution):
                    par = numpyro.sample(numpyro_name, val)
                else:
                    par = val
                kwargs[arg] = wrapper(par)

            dists[coord_name] = Distribution(**kwargs)

        return {k: dists[k] for k in self.coord_distributions}

    def _make_conditional_data(
        self, data: dict[str, ArrayLike]
    ) -> dict[CoordinateName, dict]:
        conditional_data: dict[CoordinateName, dict] = {}
        for coord_name in self.coord_distributions:
            data_map = self.conditional_data.get(coord_name, {})

            conditional_data[coord_name] = {}
            for key, val in data_map.items():
                # NOTE: behavior - if key is missing from data, we pass None
                conditional_data[coord_name][key] = get_coord_from_data_dict(val, data)

        return conditional_data

    def _make_sample_order(self) -> list[CoordinateName]:
        sample_order = []

        conditional_data = {
            k: list(set(v.values())) for k, v in self.conditional_data.items()
        }

        # First, any coord or coord pair not in conditional_data can be done first:
        for coord_name in self.coord_distributions:
            if coord_name not in self.conditional_data:
                sample_order.append(coord_name)
                conditional_data.pop(coord_name, None)

        for _ in range(128):  # NOTE: max 128 iterations
            flat_sample_order = list(
                chain(*[(s,) if isinstance(s, str) else s for s in sample_order])
            )
            for coord_name, dependencies in conditional_data.items():
                if all(dep in flat_sample_order for dep in dependencies):
                    sample_order.append(coord_name)
                    conditional_data.pop(coord_name)
                    break

            if len(conditional_data) == 0:
                break

        else:
            msg = "Circular dependency likely in conditional_data"
            raise ValueError(msg)

        return sample_order

    def __call__(
        self, data: dict[str, ArrayLike], err: dict[str, ArrayLike] | None = None
    ) -> None:
        """
        This sets up the model component in numpyro.
        """
        if err is None:
            err = {}

        dists = self.make_dists()
        for coord_name, dist_ in dists.items():
            if isinstance(coord_name, tuple):
                _data = jnp.stack([data[k] for k in coord_name], axis=-1)
                _data_err = None  # TODO: we don't support errors for joint coordinates
            else:
                _data = jnp.asarray(data[coord_name])
                _data_err = err.get(coord_name, None)

            numpyro_name = self._make_numpyro_name(coord_name)
            if _data_err is not None:
                sample_shape = (_data.shape[0],) if dist_.batch_shape == () else ()
                model_val = numpyro.sample(
                    f"{numpyro_name}:modeldata", dist_, sample_shape=sample_shape
                )

                with numpyro.plate('data', _data.shape[0]):
                    numpyro.sample(f"{numpyro_name}-obs",dist.Normal(model_val, _data_err),obs=_data,
            )
            else:
                numpyro.sample(f"{numpyro_name}-obs", dist_, obs=_data)

            # TODO: what to do if user wants to model number density?
            # Compute the log of the effective volume integral, used in the poisson
            # process likelihood
            # ln_n = obj.ln_number_density(data)
            # numpyro.factor(f"{cls.name}-factor-V", -obj.get_N())
            # numpyro.factor(f"{cls.name}-factor-ln_n", ln_n.sum())
            # numpyro.factor(f"{cls.name}-factor-extra_prior", obj.extra_ln_prior(pars))

    def sample(
        self,
        key: jax._src.random.KeyArray,
        sample_shape: Any = (),
        pars: dict[CoordinateName, Any] | None = None,
        dists: dict[CoordinateName, dist.Distribution] | None = None,
    ) -> dict[CoordinateName, jax.Array]:
        """
        Sample from the model component. If no parameters `pars` are passed, this will
        sample from the prior. All of the coordinate distributions must be sample-able
        in order for this to work.

        Parameters
        ----------
        key
            A JAX random key.
        sample_shape (optional)
            The shape of the samples to draw.
        pars (optional)
            A dictionary of parameters for the model component.
        """
        if pars is None:
            dists_ = seed(self.make_dists, key)(dists=dists)
        else:
            dists_ = self.make_dists(pars=pars, dists=dists)

        keys = jax.random.split(key, len(self.coord_distributions))

        samples: dict[CoordinateName, jax.Array] = {}
        for coord_name, key_ in zip(self._sample_order, keys, strict=True):
            extra_data = self._make_conditional_data(samples)
            shape = sample_shape if len(extra_data[coord_name]) == 0 else ()
            samples[coord_name] = dists_[coord_name].sample(
                key_, shape, **extra_data[coord_name]
            )

        return {k: samples[k] for k in self.coord_distributions}

    def evaluate_on_2d_grids(
        self,
        pars: dict[str, Any],
        grids: dict[str, ArrayLike],
        grid_coord_names: list[tuple[str, str]] | None = None,
        x_coord_name: str | None = None,
        dists: dict[CoordinateName, dist.Distribution] | None = None,
    ):
        """
        Evaluate the log-density of the model on 2D grids of coordinates paired with the
        same x coordinate. For example, in the context of a stream model, the x
        coordinate would likely be the "phi1" coordinate.

        Parameters
        ----------
        pars
            A dictionary of parameter values for the model component.
        grids
            A dictionary of 1D grids for each coordinate in the model component. The
            keys should be the names of the coordinates you want to evaluate the model
            on, and must always contain the x coordinate.
        grid_coord_names
            A list of tuples of coordinate names to evaluate the model on. The default
            is to pair the x coordinate with each other coordinate in the model
            component. For example, if the model component has coordinates "phi1",
            "phi2", and "pm1", the default grid_coord_names would be [("phi1", "phi2"),
            ("phi1", "pm1")].
        x_coord_name
            The name of the x coordinate to use for evaluating the model. If None, the
            default x coordinate will be used, which is taken to be the 0th coordinate
            name in the specified "coord_distributions".
        """
        x_coord_name = self.default_x_coord if x_coord_name is None else x_coord_name

        if x_coord_name not in self._coord_names:
            msg = f"{x_coord_name} is not a valid coordinate name"
            raise ValueError(msg)

        if grid_coord_names is None:
            # Pair the x coordinate with each other coordinate in the model component
            grid_coord_names = [
                (x_coord_name, coord_name) for coord_name in self._coord_names[1:]
            ]

        for name_pair in grid_coord_names:
            if name_pair[0] != x_coord_name:
                # TODO: we could make this more general, but then some logic below needs
                # to become more general
                msg = (
                    "We currently only support evaluating on 2D grids with the same x "
                    "coordinate axis for all grids"
                )
                raise ValueError(msg)

        # validate grid_coord_names
        for name_pair in grid_coord_names:
            for name in name_pair:
                if name not in self._coord_names:
                    msg = f"{name} is not a valid coordinate name"
                    raise ValueError(msg)

        grids_2d = self._get_grids_2d(grids, grid_coord_names)

        # Extra data to pass to log_prob() for each coordinate:
        grid_cs = {k: 0.5 * (grids[k][:-1] + grids[k][1:]) for k in grids}
        conditional_data = self._make_conditional_data(grid_cs)

        # Make the distributions for each coordinate:
        dists = self.make_dists(pars=pars, dists=dists)

        # First we have to check if the model component for the x coordinate is in a
        # joint distribution with another coordinate. If it is, we need to evaluate the
        # joint distribution on the grid and compute the marginal distribution for x:
        if x_coord_name not in self.coord_distributions:
            # At this point, x_coord_name is definitely a valid coord name, but it
            # doesn't exist as a string key in coord_distributions - it must be in a
            # joint:
            for x_joint_name_pair in self.coord_distributions:
                if x_coord_name in x_joint_name_pair:
                    break

            # Evaluate the joint distribution on the grids:
            grid1, grid2 = grids_2d[x_joint_name_pair]
            grid1_c = 0.5 * (grid1[:-1, :-1] + grid1[1:, 1:])
            grid2_c = 0.5 * (grid2[:-1, :-1] + grid2[1:, 1:])

            ln_p = dists[x_joint_name_pair].log_prob(
                jnp.stack((grid1_c, grid2_c), axis=-1),
                **conditional_data[x_joint_name_pair],
            )

            # Integrates over the other coordinate to get the marginal distribution for
            # x:
            ln_p_x = ln_simpson(ln_p, grid2_c, axis=0)
            x_grid = grid1_c

        else:
            # Otherwise, we can just evaluate the model on the x coordinate grid:
            grid = grids[x_coord_name]
            x_grid = 0.5 * (grid[:-1] + grid[1:])
            ln_p_x = dists[x_coord_name].log_prob(
                x_grid, **conditional_data[x_coord_name]
            )

        evals = {}
        for name_pair in grid_coord_names:
            grid1, grid2 = grids_2d[name_pair]

            # grid edges passed in, but we evaluate at grid centers:
            grid1_c = 0.5 * (grid1[:-1, :-1] + grid1[1:, 1:])
            grid2_c = 0.5 * (grid2[:-1, :-1] + grid2[1:, 1:])

            # Evaluate the model on the grid
            if name_pair in self.coord_distributions:
                # It's a joint distribution:
                evals[name_pair] = dists[name_pair].log_prob(
                    jnp.stack((grid1_c, grid2_c), axis=-1),
                    **conditional_data[name_pair],
                )
            else:
                # It's an independent distribution from the x_coord_name:
                ln_p_y = dists[name_pair[1]].log_prob(
                    grid2_c, **conditional_data[name_pair[1]]
                )
                evals[name_pair] = ln_p_x + ln_p_y

        return grids_2d, evals

    ###################################################################################
    # Methods that can be overridden in subclasses:
    #
    def extra_ln_prior(self, pars: dict[str, Any]):
        """
        A log-prior to add to the total log-probability. This is useful for adding
        custom priors or regularizations that are not part of the model components.
        """
        return 0.0


class ComponentMixtureModel(eqx.Module, ModelMixin):
    """
    Creating a mixture model from multiple ModelComponent objects.

    :param mixing_probs:
        the distribution of the mixing probabilities for the components in the mixture model
    :type mixing_probs: dist.Dirichlet | ArrayLike

    :param components:
        a list of the model components that make up the mixture model
    :type components: list[ModelComponent]

    :param tied_coordinates:
        (optional) A dictionary of tied coordinates, where a key should be the name of a model
        component in the mixture, and the value should be a dictionary with keys as
        the names of the coordinates in the model component and values as the names
        of the other model component to tie that coordinate to. For example,
        tied_coordinates={"offtrack": {"pm1": "stream"}} means that for the model
        component named "offtrack", use the "pm1" coordinate from the "stream" model
        component. default=None
    :type tied_coordinates: dict[str, dict[str, str]]

    :attr:`coord_names`
        the names of the component parameters
        (the keys in `coord_distributions` and `coord_parameters` of each individual component).
        Every component must have the same coordinate names so they can be combined
    :type coord_names: tuple[str]

    :attr:`_tied_order`:
        Based on which coordinates are tied, the order in which the component distributions should be modeled.
        First, components with no dependencies are modeled, then components with dependencies are modeled.
    :type _tied_order: list[str]

    :attr:`_components`:
        a dictionary of the model components, where the keys are the names of the components and the values are the components themselves.
        This is just a restructuring of the input list of components into a dictionary for easier access.
    :type _components: dict[str, ModelComponent]
    """
    mixing_probs: dist.Dirichlet | ArrayLike
    components: list[ModelComponent]
    tied_coordinates: dict[str, dict[str, str]] = eqx.field(default=None)

    coord_names: tuple[str] = eqx.field(init=False)
    _tied_order: list[str] = eqx.field(init=False)
    _components: dict[str, ModelComponent] = eqx.field(init=False)

    def __post_init__(self):
        # Some validation of the input bits:
        coord_names = None
        for component in self.components:
            if not isinstance(component, ModelComponent):
                msg = "All components must be instances of ModelComponent"
                raise ValueError(msg)

            if coord_names is None:
                coord_names = tuple(component.coord_names)
            elif tuple(component.coord_names) != coord_names:
                msg = "All components must have the same coordinate names"
                raise ValueError(msg)
        self.coord_names = coord_names

        if len({component.name for component in self.components}) != len(
            self.components
        ):
            msg = "All components must have unique names"
            raise ValueError(msg)
        self._components = {component.name: component for component in self.components}

        mix_shape = (
            self.mixing_probs.event_shape[0]
            if isinstance(self.mixing_probs, dist.Dirichlet)
            else self.mixing_probs.shape[0]
        )

        if mix_shape != len(self.components):
            msg = (
                "The mixing distribution must have the same number of components as "
                "the model."
            )
            raise ValueError(msg)

        # Validate tied coordinates:
        self.tied_coordinates = (
            self.tied_coordinates if self.tied_coordinates is not None else {}
        )
        # TODO: out of laziness, we only support non-joint coordinates in tied
        # coordinates for now...
        for component_name, coords in self.tied_coordinates.items():
            if component_name not in self._components:
                msg = (
                    f"Component '{component_name}' passed in to tied_coordinates not "
                    "found in the mixture model"
                )
                raise ValueError(msg)

            for coord_name in coords:
                if not isinstance(coord_name, str):
                    msg = "Only non-joint coordinates are supported in tied coordinates"
                    raise NotImplementedError(msg)

        # Check for circular dependencies and set up order of components to create dists
        # for:
        self._tied_order = self._make_tied_order(self.tied_coordinates)

    @property
    def component_names(self) -> tuple[str, ...]:
        return tuple(self._components.keys())

    def _make_tied_order(
        self, tied_coordinates: dict[str, dict[str, str]]
    ) -> list[str]:
        """
        Parameters
        ----------
        tied_coordinates
            A dictionary of tied coordinates, where a key should be the name of a model
            component in the mixture, and the value should be a dictionary with keys as
            the names of the coordinates in the model component and values as the names
            of the other model component to tie that coordinate to. For example,
            tied_coordinates={"offtrack": {"pm1": "stream}} means that for the model
            component named "offtrack", use the "pm1" coordinate from the "stream" model
            component.
        """
        tied_coordinates = copy.deepcopy(tied_coordinates)
        dependencies = {k: list(set(v.values())) for k, v in tied_coordinates.items()}

        tied_order = []

        # First, any component with no dependencies can be done first:
        for name in self.component_names:
            if name not in dependencies:
                tied_order.append(name)

        for _ in range(128):  # NOTE: max 128 iterations, arbitrary
            for name in tied_coordinates:
                if all(dep in tied_order for dep in dependencies[name]):
                    tied_order.append(name)
                    tied_coordinates.pop(name, None)
                    break

            if len(tied_coordinates) == 0:
                break

        else:
            msg = "Circular dependency likely in tied_coordinates"
            raise ValueError(msg)

        return tied_order

    def _make_concatenated(self) -> dict[str, Any]:
        # Deal with tied coordinates here across components:
        concatenated: dict[str, ConcatenatedDistributions] = {}

        # TODO: figure out batch shape and event shape based on all components, and pass
        # that in to the ConcatenatedDistributions so that all concatenated have the
        # same shape expectations!

        all_dists: dict[str, dict[str, dist.Distribution]] = defaultdict(dict)
        for component_name in self._tied_order:
            component = self._components[component_name]
            tied_map = self.tied_coordinates.get(component_name, {})

            # TODO: ._dists is a list now, so this won't work...
            override_dists = {
                override_coord: all_dists[dep][override_coord]
                for override_coord, dep in tied_map.items()
            }

            ## Having forced the offtrack proper motions (for example) to be identical
            ##  to the stream proper motions, make_dists will create the distributions
            ##  including numpyro.sample. This should be equivalent to creating an if
            ##  statement where for tied coordinates (or fixed coordinates),
            ##  we can get the numpyro_name here and do numpyro.deterministic for that parameter.
            ##  This adds the flexibility of being able to have "loosely tied" coordinates
            ##  i.e. where the tied coordinates are not identical, but are related in some way.

            dists = component.make_dists(dists=override_dists)
            concatenated[component_name] = ConcatenatedDistributions(
                dists=list(dists.values())
            )
            all_dists[component_name] = dists

        return concatenated

    def __getitem__(self, key: str) -> ModelComponent:
        return self._components[key]

    def __call__(
        self, data: dict[str, ArrayLike], err: dict[str, ArrayLike] | None = None
    ) -> None:
        """
        This sets up the mixture model in numpyro.
        """
        probs = numpyro.sample("mixture-probs", self.mixing_probs)

        concatenated = self._make_concatenated()

        # Here we make sure the order we pass in the components respects the original
        # order the user passed in, for interpretation of the probabilities:
        mixture = dist.MixtureGeneral(
            dist.Categorical(probs),
            [concatenated[name] for name in self.component_names],
        )

        stacked_data = jnp.stack([data[k] for k in self.coord_names], axis=-1)
        if err is None:
            numpyro.sample("mixture", mixture, obs=stacked_data)
        else:
            # TODO: this is a hack to make sure we have errors for all coordinates
            #        Do we need to have errors for all coordinates???
            # NOTE: we should warn the user that this is happening...
            # err = {k: err.get(k, jnp.full(stacked_data.shape[0], 1e-8)) for k in data}
            err = {k: err.get(k, 1e-4) for k in data}
            sample_shape = (stacked_data.shape[0],) if mixture.batch_shape == () else ()
            model_data = numpyro.sample("mixture:modeldata", mixture, sample_shape=sample_shape)

            # TODO: something weird here. How are joint coordinates handled? coord_names
            # I think it just the names of the coordinates, not the component names. So
            # can they ever have size 2, as below? Maybe this isn't then the source of
            # the issue...

            i = 0
            for name in self.coord_names:
                # TODO: assumes that joints can only have at most 2 coordinates. This is
                # probably also implicitly assumed elsewhere, so need to enforce this at
                # some validation time
                size = 2 if isinstance(name, tuple) else 1
                slc = slice(i, i + size)

                with numpyro.plate('data', stacked_data.shape[0]):
                    if size == 1:
                        model_data_dist = dist.Normal(jnp.squeeze(model_data[:, slc]), err[name])
                        numpyro.sample(f"{name}-obs", model_data_dist, obs=jnp.squeeze(stacked_data[:, slc]))
                    elif size == 2:
                        model_data_dist = dist.Normal(model_data[:, slc], err[name])
                        numpyro.sample(f"{name}-obs", model_data_dist, obs=stacked_data[:, slc])
                i += size

    def pack_params(
        self, pars: dict[str, dict[CoordinateName, dict]]
    ) -> dict[str, Any]:
        """
        Convert a dictionary of parameters as a nested dictionary into a flat dictionary
        with packed numpyro-compatible names.

        Parameters
        ----------
        pars
            A dictionary of parameters where the keys are the names of the components in
            the model and the values are dictionaries of the parameters for each
            coordinate in the component.
        """
        packed_pars = {}
        for component_name, component_pars in pars.items():
            packed_pars.update(
                self._components[component_name].pack_params(component_pars)
            )
        return packed_pars

    def expand_numpyro_params(
        self, pars: dict[str, Any], skip_invalid: bool = False
    ) -> dict[str, dict[CoordinateName, Any]]:
        """
        Convert a dictionary of numpyro parameters into a nested dictionary where the
        keys are the coordinate names and parameter name.

        Parameters
        ----------
        pars
            A dictionary of numpyro parameters where the keys are the names of the
            parameters created with numpyro.sample().
        """
        pars = copy.deepcopy(pars)

        expanded_pars: dict[str, dict] = {}
        for component_name in self._tied_order:
            component = self._components[component_name]
            tied_map = self.tied_coordinates.get(component.name, {})
            component_pars = {}
            for key in list(pars.keys()):  # convert to list because we change dict keys
                if key.startswith(f"{component.name}:"):
                    component_pars[key] = pars.pop(key)
            expanded_pars[component.name] = component.expand_numpyro_params(
                component_pars, skip_invalid=skip_invalid
            )

            # TODO(adrn): duplicated code
            for override_coord, dep in tied_map.items():
                # First, we set with the parameters from the coord_parameters, then
                # update values from the tied component
                expanded_pars[component.name][override_coord] = copy.deepcopy(
                    self._components[dep].coord_parameters[override_coord]
                )
                expanded_pars[component.name][override_coord].update(
                    expanded_pars[dep][override_coord]
                )

        expanded_pars.update(pars)
        return expanded_pars

    def evaluate_on_2d_grids(
        self,
        pars: dict[str, Any],
        grids: dict[str, ArrayLike],
        grid_coord_names: list[tuple[str, str]] | None = None,
        x_coord_name: str | None = None,
    ):
        # Crude way of detecting if pars are already expanded or not:
        if all(name in pars for name in self.component_names):
            expanded_pars = pars
        else:
            expanded_pars = self.expand_numpyro_params(pars)

        # Deal with tied coordinates here across components:
        # TODO(adrn): duplicated code
        component_dists: dict[str, dict[CoordinateName, dist.Distribution]] = {}
        dists: dict[str, dict] = {}
        for component_name in self._tied_order:
            tied_map = self.tied_coordinates.get(component_name, {})

            dists[component_name] = {
                override_coord: component_dists[dep][override_coord]
                for override_coord, dep in tied_map.items()
            }

            component_dists[component_name] = self._components[
                component_name
            ].make_dists(
                pars=expanded_pars[component_name],
                dists=dists.get(component_name, {}),
            )

        terms: dict[str, list[jax.Array]] = {}
        for component in self.components:
            # TODO: need to override dists here too, but damn that interface sucks
            all_grids, component_terms = component.evaluate_on_2d_grids(
                pars=expanded_pars[component.name],
                grids=grids,
                grid_coord_names=grid_coord_names,
                x_coord_name=x_coord_name,
                dists=dists.get(component.name, {}),
            )
            for k, v in component_terms.items():
                if k not in terms:
                    terms[k] = []
                terms[k].append(v)

        # use "mixture-probs" to weight the component terms
        terms = {
            k: jax.scipy.special.logsumexp(
                jnp.array(v).T, axis=-1, b=pars["mixture-probs"]
            ).T
            for k, v in terms.items()
        }
        return all_grids, terms
