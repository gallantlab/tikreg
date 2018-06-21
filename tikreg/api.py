'''
http://scikit-learn.org/stable/developers/contributing.html#rolling-your-own-estimator
'''


###
estimator = obj.fit(data, targets)
estimator = obj.fit(data)

estimator.fit(X, y)


###
prediction = obj.predict(data)

###
score = obj.score(data)

class BandedRidgeKernel(object):
    def __init__(self, spatial_hyparams=(-2, 10), param1=1, param2=2):
        self.param1 = param1
        self.param2 = param2

    def fit(self, Xtrains, Ytrain, temporal_prior='hrf'):
        '''
        '''
        self.weights_dual_ = None
        self.weights_primal_ = None
        return self

    def predict(self, Xtests, **kwargs):
        Yhat = np.dot(Xtests, self.weights_primal_)
        return Yhat

    def score(self, Ytrue, Ypred, **kwargs):
        return np.sum((Ytrue - Ypred)**2.0, axis=0)
