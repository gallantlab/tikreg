# tikreg: Tikhonov regression in Python

[![Build Status](https://travis-ci.com/gallantlab/tikreg.svg?token=DG1xpt4Upohy9kdU6zzg&branch=master)](https://travis-ci.com/gallantlab/tikreg)

**tikreg** is a Python package that efficiently implements Tikhonov regression.

Tikhonov regression gives us a framework to estimate spatiotemporal encoding models with non-spherical multivariate normal priors. This framework is useful to model biological signals. This package was developed to analyze brain data collected using functional magnetic resonance imaging (fMRI). `tikreg`  can also be used to model other neuroimaging signals (e.g. 2P, ECoG, etc) and LTI signals more generally.

## Advantages
* Useful when building large joint models that combine multiple feature spaces
* Transfer function estimation via regularized FIR models
* Efficient implementation of ridge regression for multiple outputs
* Dual and primal solutions for ridge regression

## Installation
Clone the repo from GitHub and do the usual python install from the command line

```
$ git clone https://github.com/gallantlab/tikreg.git
$ cd tikreg
$ sudo python setup.py install
```

## Documentation

APPOLOGIES FOR THE LACK OF DOCUMENTATION!

Working on it! In the mean time, please refer to the unitests.

## Cite as
Nunez-Elizalde AO, Huth AG, and Gallant, JL (2018). Voxelwise encoding models with non-spherical multivariate normal priors. BioRxiv. https://doi.org/10.1101/386318.

