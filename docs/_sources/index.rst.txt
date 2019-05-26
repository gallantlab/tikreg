.. tikreg documentation master file, created by
   sphinx-quickstart on Sat May 25 17:01:59 2019.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

tikreg: Tikhonov regression in Python
=====================================



What is **tikreg**?
*******************

tikreg is a Python package that efficiently implements Tikhonov regression.

Tikhonov regression can be used to estimate encoding models with non-spherical multivariate normal priors. This framework is useful to model biological signals. This package was developed to analyze brain data collected using functional magnetic resonance imaging (fMRI). tikreg can also be used to model other neuroimaging signals (e.g. 2P, ECoG, etc) and LTI signals more generally.

Advantages
**********

* Useful when building large joint models that combine multiple feature spaces
* Transfer function estimation via regularized FIR models
* Efficient implementation of ridge regression for multiple outputs
* Dual and primal solutions for ridge regression


.. toctree::
   :maxdepth: 2
   :caption: Contents:



Learn more:
===========

.. toctree::
   :maxdepth: 2

   api/index.rst


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
