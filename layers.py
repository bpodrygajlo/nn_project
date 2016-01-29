import numpy
import theano
import numpy as np
import theano.tensor as T
#
# These are taken from https://github.com/mila-udem/blocks
# 

class Constant():
    """Initialize parameters to a constant.
    The constant may be a scalar or a :class:`~numpy.ndarray` of any shape
    that is broadcastable with the requested parameter arrays.
    Parameters
    ----------
    constant : :class:`~numpy.ndarray`
        The initialization value to use. Must be a scalar or an ndarray (or
        compatible object, such as a nested list) that has a shape that is
        broadcastable with any shape requested by `initialize`.
    """
    def __init__(self, constant):
        self._constant = numpy.asarray(constant)

    def generate(self, rng, shape):
        dest = numpy.empty(shape, dtype=np.float32)
        dest[...] = self._constant
        return dest


class IsotropicGaussian():
    """Initialize parameters from an isotropic Gaussian distribution.
    Parameters
    ----------
    std : float, optional
        The standard deviation of the Gaussian distribution. Defaults to 1.
    mean : float, optional
        The mean of the Gaussian distribution. Defaults to 0
    Notes
    -----
    Be careful: the standard deviation goes first and the mean goes
    second!
    """
    def __init__(self, std=1, mean=0):
        self._mean = mean
        self._std = std

    def generate(self, rng, shape):
        m = rng.normal(self._mean, self._std, size=shape)
        return m.astype(np.float32)


class Uniform():
    """Initialize parameters from a uniform distribution.
    Parameters
    ----------
    mean : float, optional
        The mean of the uniform distribution (i.e. the center of mass for
        the density function); Defaults to 0.
    width : float, optional
        One way of specifying the range of the uniform distribution. The
        support will be [mean - width/2, mean + width/2]. **Exactly one**
        of `width` or `std` must be specified.
    std : float, optional
        An alternative method of specifying the range of the uniform
        distribution. Chooses the width of the uniform such that random
        variates will have a desired standard deviation. **Exactly one** of
        `width` or `std` must be specified.
    """
    def __init__(self, mean=0., width=None, std=None):
        if (width is not None) == (std is not None):
            raise ValueError("must specify width or std, "
                             "but not both")
        if std is not None:
            # Variance of a uniform is 1/12 * width^2
            self._width = numpy.sqrt(12) * std
        else:
            self._width = width
        self._mean = mean

    def generate(self, rng, shape):
        w = self._width / 2
        m = rng.uniform(self._mean - w, self._mean + w, size=shape)
        return m.astype(np.float32)


class Layer(object):
    def __init__(self, name=None, rng=None):
        if rng is None:
            rng = T.shared_randomstreams.RandomStreams(numpy.random.randint(9999999))
        self.rng = rng
        self.name = name
    
    def init_parameters(self):
        pass
    
    def get_parameters(self):
        return []
        
        
class ConvLayer(Layer):
    def __init__(self, image_stack, num_filters, filter_size=(5,5), weight_init=IsotropicGaussian(0.05), bias_init=Constant(0.0), **kwargs):
        super(ConvLayer, self).__init__(**kwargs)
        self.weight_init = weight_init
        self.bias_init = bias_init
        self.num_filters = num_filters
        self.filter_size = filter_size
        self.image_stack = image_stack
        self.CW1 = None
        self.CB1 = None

    def init_parameters(self):
        self.CW1 = theano.shared(np.zeros((self.num_filters,self.image_stack) + self.filter_size, dtype='float32'), name = 'conv_weight')
        self.CW1.tag.initializer = self.weight_init
        
        self.CB1 = theano.shared(np.zeros((self.num_filters,), dtype='float32') , name = 'conv_bias')
        self.CB1.tag.initializer = self.bias_init
        
    def get_parameters(self):
        return [self.CW1, self.CB1]

    def fprop(self, X, printing = False):
        Ret = T.maximum(0.0, T.nnet.conv2d(X, self.CW1) + self.CB1.dimshuffle('x',0,'x','x'))
        if printing:
            print "ConvLayer"
            print "Layer shape: %s" % (Ret.tag.test_value.shape,)
        return Ret
        
        
        
class DropoutLayer(Layer):
    def __init__(self, droprate = 0.2, **kwargs):
        super(DropoutLayer, self).__init__(**kwargs)
        self.droprate = droprate
        
    def fprop(self, X, printing = False):
        mask = self.rng.binomial(n=1,p=1.0-self.droprate,size=X.shape)
        Ret = X * T.cast(mask, dtype='float32')
        if printing:
            print "DropoutLayer"
            print "Layer shape: %s" % (Ret.tag.test_value.shape,)
        return Ret

class TanhLayer(Layer):
    def __init__(self,  **kwargs):
        super(TanhLayer, self).__init__(**kwargs)
    
    def fprop(self, X, printing=False):
        Ret = T.tanh(X)
        if printing:
            print "TanhLayer"
            print "Layer shape: %s" % (Ret.tag.test_value.shape,)
        return Ret

    
class ReLULayer(Layer):
    def __init__(self, **kwargs):
        super(ReLULayer, self).__init__(**kwargs)
    
    def fprop(self, X, printing =False):
        Ret = T.switch(X<0.0,0.0,X)
        if printing:
            print "ReLULayer"
            print "Layer shape: %s" % (Ret.tag.test_value.shape,)
        return Ret
    
    
class PoolingLayer(Layer):
    def __init__(self, shape = (2,2), **kwargs):
        self.shape = shape
    
    def fprop(self, X , printing = False):
        Ret = T.signal.downsample.max_pool_2d(X, self.shape, ignore_border=True)
        if printing:
            print "PoolingLayer "
            print "Layer shape: %s" % (Ret.tag.test_value.shape,)
        return Ret

    
class SoftMaxLayer(Layer):
    def __init__(self, **kwargs):
        super(SoftMaxLayer, self).__init__(**kwargs)
    
    def fprop(self, X, printing = False):
        Ret = T.nnet.softmax(X)
        if printing:
            print "SoftMaxLayer "
            print "Layer shape: %s" % (Ret.tag.test_value.shape,)
        return Ret
    

class FlattenLayer(Layer):
    def __init__(self,**kwargs):
        super(FlattenLayer,self).__init__(**kwargs)
        
    def fprop(self,X, printing = False):
        Ret = X.flatten(2)
        if printing:
            print "FlattenLayer "
            print "Layer shape: %s" % (Ret.tag.test_value.shape,)
        return Ret
    
class FullyConnectedLayer(Layer):
    def __init__(self, num_in, num_out, weight_init=IsotropicGaussian(0.05), bias_init=Constant(0.0), **kwargs):
        super(FullyConnectedLayer,self).__init__(**kwargs)
        
        self.weight_init =weight_init
        self.bias_init = bias_init
        
        self.num_in = num_in
        self.num_out = num_out
        
        self.FW1 = None
        self.FB1 = None


    def init_parameters(self):
        self.FW1 = theano.shared(np.zeros((self.num_in, self.num_out), dtype='float32') , name = 'weight')
        self.FW1.tag.initializer = self.weight_init
        
        self.FB1 = theano.shared(np.zeros((self.num_out,), dtype='float32'), name = 'bias')
        self.FB1.tag.initializer = self.bias_init
        
    
    def get_parameters(self):
        return [self.FW1, self.FB1]

    def fprop(self, X, printing = False):
        Ret = T.maximum(0.0, 
                        T.dot(X, self.FW1) + self.FB1.dimshuffle('x',0))
        if printing:
            print "FullyConnectedLayer "
            print "Layer shape: %s" % (Ret.tag.test_value.shape,)
        return Ret

class InputLayer(Layer):
    def __init__(self, **kwargs):
        super(InputLayer,self).__init__(**kwargs)
    
    def fprop(self,X, printing = False):
        if printing:
            print "InputLayer"
            print "Layer shape: %s" % (X.tag.test_value.shape,)
        return X
        

class ConvNet(object):
    def __init__(self, name ="unnamed", desc = "no description", layers=None):
        if layers is None:
            layers = []
        self.layers = layers
        self.name = name
        self.desc = desc
    
    def add(self, layer):
        self.layers.append(layer)
    
    def init_parameters(self):
        params = []
        for layer in self.layers:
            layer.init_parameters()
            params += layer.get_parameters()
        return params
    
    def fprop(self, X, printing = False):
        for layer in self.layers:
            X = layer.fprop(X, printing = printing)
        return X
        
        



