.. _intalg-label:

======================
Integration algorithms
======================

This page lists the integration algorithms currently implemented.

.. _vegas-label:

Vegas
=====

This implementation of the Vegas algorithm closely follow the description of the importance sampling in the original `Vegas <https://doi.org/10.1016/0021-9991(78)90004-9>`_ paper.


.. autoclass:: vegasflow.vflow.VegasFlow
    :members: freeze_grid, unfreeze_grid, refine_grid, run_event

