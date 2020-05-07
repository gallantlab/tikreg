# tikreg: Tikhonov regression in Python

[![Build Status](https://travis-ci.com/gallantlab/tikreg.svg?token=DG1xpt4Upohy9kdU6zzg&branch=master)](https://travis-ci.com/gallantlab/tikreg)
[![License](https://img.shields.io/badge/license-BSD%203--Clause-blue)](https://opensource.org/licenses/BSD-3-Clause)
[![codecov](https://codecov.io/gh/gallantlab/tikreg/branch/master/graph/badge.svg)](https://codecov.io/gh/gallantlab/tikreg)
[![Downloads](https://pepy.tech/badge/tikreg)](https://pepy.tech/project/tikreg)

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

Or with pip:

```
$ pip install tikreg
```

## Getting started

### Cross-validated ridge regression

The following code performs 5-fold cross-validated ridge regression. 
  
```python
  from tikreg import models
  from tikreg import utils as tikutils
  
  # Generate synthetic data
  weights_true, (Xtrain, Xtest), (Ytrain, Ytest) = tikutils.generate_data(noise=1, testsize=100)
  
  # Specify fit
  options = dict(ridges=np.logspace(0,3,11), weights=True, metric='rsquared')
  fit = models.cvridge(Xtrain, Ytrain, Xtest, Ytest, **options)
  
  # Evaluate results
  weights_estimate = fit['weights']
  weights_corr = tikutils.columnwise_correlation(weights_true, weights_estimate)
  print(weights_corr.mean())  	# > 0.9
  print(fit['cvresults'].shape) # (5, 1, 11, 2): (nfolds, 1, nridges, nresponses)
```

By default, the optimal ridge regularization parameter is found across the population of responses. We can specifiy that the optimal regularization parameter be found for each individual response (`population_optimum=False`). 
  
```python
  options = dict(ridges=np.logspace(0,3,11), performance=True, predictions=True, weights=True, metric='rsquared')
  fit = models.cvridge(Xtrain, Ytrain, Xtest, Ytest, population_optimum=False, **options)
  print(fit.keys())
```

The model performance (`performance=True`) and the test set predictions (`predictions=True`) are also computed in the example above. 

Conveniently, `tikreg.models.cvridge()` will choose an efficient method to fit the ridge regression model automatically. Whenever the number of features is greater than the number of samples (*P > N*), the fit will be performed using kernel ridge regression. 

```python
  # P >> N
  nfeatures, ntimepoints = 1000, 100
  weights_true, (Xtrain, Xtest), (Ytrain, Ytest) = tikutils.generate_data(n=ntimepoints, p=nfeatures, testsize=100)

  # Model is automatically fit using kernel ridge
  options = dict(ridges=np.logspace(0,3,11), weights=True, metric='rsquared')
  fit = models.cvridge(Xtrain, Ytrain, Xtest, Ytest, **options)
  
  # The weights are in the feature space
  weights_estimate = fit['weights']
  print(weights.shape)         # (nfeatures, nresponses)
```


## Documentation

https://gallantlab.github.io/tikreg/index.html

## Tutorials

moar tutorials coming soon.

### Non-spherical MVN prior on features 
https://nbviewer.jupyter.org/github/gallantlab/tikreg/blob/master/examples/tutorial_feature_priors.ipynb

[Launch Google Colab notebook](https://colab.research.google.com/github/gallantlab/tikreg/blob/master/examples/tutorial_feature_priors.ipynb)

### Non-spherical MVN prior on temporal delays
https://nbviewer.jupyter.org/github/gallantlab/tikreg/blob/master/examples/tutorial_temporal_priors.ipynb

[Launch Google Colab notebook](https://colab.research.google.com/github/gallantlab/tikreg/blob/master/examples/tutorial_temporal_priors.ipynb)



### Banded ridge regression:
https://nbviewer.jupyter.org/github/gallantlab/tikreg/blob/master/examples/tutorial_banded_ridge_polar.ipynb


## Cite as
Nunez-Elizalde AO, Huth AG, and Gallant, JL (2019). Voxelwise encoding models with non-spherical multivariate normal priors. NeuroImage. https://doi.org/10.1016/j.neuroimage.2019.04.012

