import numpy as np
from fuel.datasets.cifar10 import CIFAR10
from fuel.transformers import ScaleAndShift, Cast, Flatten, Mapping
from fuel.streams import DataStream
from fuel.schemes import SequentialScheme, ShuffledScheme

batch_size = 100

CIFAR10.default_transformers = (
    (ScaleAndShift, [2.0 / 255.0, -1], {'which_sources': 'features'}),
    (Cast, [np.float32], {'which_sources': 'features'}))

cifar_train = CIFAR10(("train",), subset=slice(None,40000))
#this stream will shuffle the MNIST set and return us batches of 100 examples
cifar_train_stream = DataStream.default_stream(
    cifar_train,
    iteration_scheme=ShuffledScheme(cifar_train.num_examples, batch_size))
                                               
cifar_validation = CIFAR10(("train",), subset=slice(40000, None))


# We will use larger portions for testing and validation
# as these dont do a backward pass and reauire less RAM.
cifar_validation_stream = DataStream.default_stream(
    cifar_validation, iteration_scheme=SequentialScheme(cifar_validation.num_examples, batch_size))
cifar_test = CIFAR10(("test",))
cifar_test_stream = DataStream.default_stream(
    cifar_test, iteration_scheme=SequentialScheme(cifar_test.num_examples, batch_size))
    
    
print "The streams return batches containing %s" % (cifar_train_stream.sources,)

print "Each trainin batch consits of a tuple containing:"
for element in next(cifar_train_stream.get_epoch_iterator()):
    print " - an array of size %s containing %s" % (element.shape, element.dtype)
    
print "Validation/test batches consits of tuples containing:"
for element in next(cifar_test_stream.get_epoch_iterator()):
    print " - an array of size %s containing %s" % (element.shape, element.dtype)
