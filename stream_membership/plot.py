import numpy as np


def _plot_projections(
    grids,
    ims,
    axes=None,
    label=True,
    pcolormesh_kwargs=None,
    coord_names=None
):
    if coord_names is None:
        coord_names = [k for k in grids.keys() if k != 'phi1']

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
            len(coord_names),
            1,
            figsize=(10, 2 + 2 * len(coord_names)),
            sharex=True,
            sharey="row",
            constrained_layout=True,
        )

    if isinstance(axes, mpl.axes.Axes):
        axes = [axes]
    axes = np.array(axes)

    for i, name in enumerate(coord_names):
        grid1, grid2 = grids[name]
        axes[i].pcolormesh(
            grid1, grid2, ims[name], shading="auto", **pcolormesh_kwargs
        )
        axes[i].set_ylim(grid2.min(), grid2.max())

        if label:
            axes[i].set_ylabel(_default_labels[name])

    axes[0].set_xlim(grid1.min(), grid1.max())

    return axes.flat[0].figure, axes


def plot_data_projections(
    data,
    grids,
    axes=None,
    label=True,
    smooth=1.0,
    pcolormesh_kwargs=None,
    coord_names=None
):
    from scipy.ndimage import gaussian_filter

    if coord_names is None and grids is None:
        raise ValueError()
    elif coord_names is None:
        coord_names = [k for k in grids.keys() if k != 'phi1']

    ims = {}
    im_grids = {}
    for name in coord_names:
        H_data, xe, ye = np.histogram2d(
            data["phi1"], data[name], bins=(grids["phi1"], grids[name])
        )
        if smooth is not None:
            H_data = gaussian_filter(H_data, smooth)
        im_grids[name] = (xe, ye)
        ims[name] = H_data.T

    return _plot_projections(
        grids=im_grids,
        ims=ims,
        axes=axes,
        label=label,
        pcolormesh_kwargs=pcolormesh_kwargs,
    )