Models
======

Introduction
------------
This module contains the classes from which all models will be created. The models are used to recreate different stream components.
Most of the models are spline-based and track stream or background properties (:math:`\phi_2, \mu_{\phi_1}, \mu_{\phi_2}`), as a function of :math:`\phi_1$`.

Generic Functions for all Models
--------------------------------
The generic functions are stored in the ``ModelMixin`` class. They include plotting functions to visualize the 
model projections (``plot_model_projections``) and the residuals relative to the data (``plot_residual_projections``).
They also include the functionality in ``_get_grids_2d`` to create a dictionary of 2D grids from a dictionary of 1D grids in each stream parameter 
(e.g. 
:math:`\phi_1,  \mu_{\phi_1}, \mu_{\phi_2}`
).
This is critical because the models are being created in two-dimensional spaces (e.g. :math:`\phi_1-\phi_2`) and 
we need to evaluate the model on a grid in those spaces.

Example
~~~~~~~
.. code-block:: python

    from stream_memberships.models import ModelMixin
    import numpy as np

    # Create a dictionary of 1D grids
    grids_1d = {'phi1': np.linspace(-10, 10, 100),
                'phi2': np.linspace(-10, 10, 100),
                'mu_phi1': np.linspace(-10, 10, 100),
                'mu_phi2': np.linspace(-10, 10, 100)}

    # Define the names of the 2D grid coordinates
    grid_coord_names = [('phi1', 'phi2'), ('phi1', 'mu_phi1'), ('phi1', 'mu_phi2')]

    # Create a dictionary of 2D grids
    grid_dict_2d = ModelMixin._get_grids_2d(grids_1d, grid_coord_names)

    # Print the keys of the 2D grid dictionary
    print(grid_dict_2d.keys())

    # Print the shape of the 2D grid dictionary
    print(grid_dict_2d['phi1'].shape)

Creating Model Components
-------------------------

The ``ModelComponent`` class is the base class for all model components. It inherits the ``ModelMixin`` and ``equinox.Module`` classes, 
the latter of which takes models and makes them `dataclasses <https://docs.python.org/3/library/dataclasses.html>`_ and 
`pytrees <https://jax.readthedocs.io/en/latest/pytrees.html>`_.
The ``ModelComponent`` class is used to create the different components of the model (e.g. ``background``, ``stream``, ``offtrack``) 
by interfacing with numpyro behind-the-scenes. 
It is also used to sample from the component's probability distribution and return the log-density of the model component at a given coordinate.

The ``ModelComponent`` class has the following attributes which it takes in as parameters:

* ``name`` \: str
            the name of the model component (usually either ``background``, ``stream``, or ``offtrack``)

* ``coord_distributions`` \: dict[str | tuple, Any]
                          a dictionary of the distributions of the component parameters. 
                          The keys are the names of the component parameters 
                          (e.g. ```'phi1'```, ``'phi2'``, ``'mu_phi1'```, ``'mu_phi2'``, ``('phi1', 'phi2')```, etc.) 

                          and the values are numpyro distributions of those parameters.
* ``coord_parameters`` \: dict[str \| tuple, dict[str, dist.Distribution \| tuple \| ArrayLike \| dict]]
                        a dictionary of the parameters of the distributions in ``coord_distributions```.
                        The keys are the names of the component parameters (the keys in ``coord_distributions```) 
                        and the values are dictionaries containing the parameters for the distributions.
                        For example, a truncated normal distribution (``dist.TruncatedNormal``` in ```numpyro```) 
                        might have the parameters ``loc``, ``scale``, ``low``, and ``high``.

* ``default_x_coord`` \: str | None = None
                       (optional) the default x-coordinate for the model component

* ``conditional_data`` \: dict[str | tuple[str, str], dict[str, str]] = eqx.field(default=None)
                        (optional) a dictionary of any additional data that is required for evaluating the
                        log-probability of a coordinate's probability distribution. For example, a
                        spline-enabled distribution might require the phi1 data to evaluate the spline
                        at the phi1 values. 
                        The keys are the names of the component parameters
                        (e.g. ``'phi1'``, ``'phi2'```, ``'mu_phi1'``, ``'mu_phi2'``, ``('phi1', 'phi2')``, etc.) 
                        and the values are dictionaries of the conditional data for those parameters.

It also has two other attributes which are defined during initialization with ``__post_init__``:

* ``_coord_names`` \: list[str] = eqx.field(init=False)`
                   (optional) the names of the component parameters (the keys in ``coord_distributions`` and ``coord_parameters``)

* ``_sample_order`` \: list[str | tuple[str, str]] = eqx.field(init=False)
                    (optional) the order in which the component parameters should be sampled




Creating a Mixture Model From Multiple Components
-------------------------------------------------

The ``ComponentMixtureModel`` class has the following attributes which it takes in as parameters:

* ``mixing_probs`` \: dist.Dirichlet | ArrayLike
                    the mixing probabilities of the model components

* ``components`` \: list[ModelComponent]
                  a list of the model components that will be mixed together to create the mixture model

* ``tied_coordinates`` \: dict[str, dict[str, str]] = eqx.field(default=None)
                        (optional) a dictionary of the tied coordinates between the model components.
                        The keys are the names of the component parameters (the keys in ``coord_distributions`` and ``coord_parameters``)
                        and the values are dictionaries of the tied coordinates for those parameters.
                        KT: FIX THIS DESCRIPTION

It also has three other attributes which are defined during initialization with `__post_init__`:

* ``coord_names`` \: tuple[str] = eqx.field(init=False)

* ``_tied_order`` \: list[str] = eqx.field(init=False)

* ``_components`` \: dict[str, ModelComponent] = eqx.field(init=False)