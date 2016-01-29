
# coding: utf-8

# In[37]:

import numpy as np
import numpy as numpy


# In[38]:

from theano import function, config, shared, sandbox
import theano.sandbox.cuda.basic_ops
import theano.tensor as T
import numpy
import time

def check_device():
    vlen = 10 * 30 * 768  # 10 x #cores x # threads per core
    iters = 1000

    rng = np.random.RandomState(22)
    x = shared(np.asarray(rng.rand(vlen), 'float32'))
    f = function([], sandbox.cuda.basic_ops.gpu_from_host(T.exp(x)))
    print(f.maker.fgraph.toposort())
    t0 = time.time()
    for i in xrange(iters):
        r = f()
    t1 = time.time()
    print("Looping %d times took %f seconds" % (iters, t1 - t0))
    print("Result is %s" % (r,))
    print("Numpy result is %s" % (np.asarray(r),))
    if np.any([isinstance(x.op, T.Elemwise) for x in f.maker.fgraph.toposort()]):
        print('Used the cpu')
    else:
        print('Used the gpu')


# In[39]:

try:
    check_device()
except EnvironmentError:
    print "Could not set up GPU"


# In[40]:

import theano
import theano.tensor as T

from data_loading import *
from layers import *


# A theano variable is an entry to the cmputational graph
# We will need to provide its value during function call
# X is batch_size x num_channels x img_rows x img_columns
X = T.tensor4('X')

# Y is 1D, it lists the targets for all examples
Y = T.matrix('Y', dtype='uint8')

#The tag values are useful during debugging the creation of Theano graphs

X_test_value, Y_test_value = next(cifar_train_stream.get_epoch_iterator())
#
# Unfortunately, test tags don't work with convolutions with newest Theano :(
#
theano.config.compute_test_value  = 'warn' # Enable the computation of test values


X.tag.test_value = X_test_value[:3]
Y.tag.test_value = Y_test_value[:3]


import sys

if len(sys.argv) < 2:
    print 'Please specify CNN file'
    sys.exit(1)

import cPickle
f = file(sys.argv[1], 'rb')
cnn = cPickle.load(f)
f.close()

print 'Loaded CNN'
print 'Name: ', cnn.name
print 'Description', cnn.desc

model_parameters = cnn.init_parameters()
log_probs = cnn.fprop(X, printing = True)

theano.config.compute_test_value = 'off'


# In[47]:

predictions = T.argmax(log_probs, axis=1)

error_rate = T.neq(predictions,Y.ravel()).mean()
nll = - T.log(log_probs[T.arange(Y.shape[0]), Y.ravel()]).mean()

weight_decay = 0.0
for p in model_parameters:
    if p.name =='weight' or p.name == 'conv_weight':
        weight_decay = weight_decay + 0.0001 * (p**2).sum()

cost = nll + weight_decay

#At this point stop computing test values
theano.config.compute_test_value = 'off' # Enable the computation of test values


# In[48]:

# The updates will update our shared values
updates = []


# In[49]:

lrate = T.scalar('lrate',dtype='float32')
momentum = T.scalar('momentum',dtype='float32')

# Theano will compute the gradients for us
gradients = theano.grad(cost, model_parameters)

#initialize storage for momentum
velocities = [theano.shared(np.zeros_like(p.get_value()), name='V_%s' %(p.name, )) for p in model_parameters]

for p,g,v in zip(model_parameters, gradients, velocities):
    v_new = momentum * v - lrate * g
    p_new = p + v_new
    updates += [(v,v_new), (p, p_new)]


# In[50]:

updates


# In[51]:

#compile theano functions

#each call to train step will make one SGD step
train_step = theano.function([X,Y,lrate,momentum],[cost, error_rate, nll, weight_decay],updates=updates)
#each call to predict will return predictions on a batch of data
predict = theano.function([X], predictions)


# In[52]:

def compute_error_rate(stream):
    errs = 0.0
    num_samples = 0.0
    for X, Y in stream.get_epoch_iterator():
        errs += (predict(X)!=Y.ravel()).sum()
        num_samples += Y.shape[0]
    return errs/num_samples


# In[53]:

#utilities to save values of parameters and to load them

def init_parameters():
    rng = np.random.RandomState(1234)
    for p in model_parameters:
        p.set_value(p.tag.initializer.generate(rng, p.get_value().shape))

def snapshot_parameters():
    return [p.get_value(borrow=False) for p in model_parameters]

def load_parameters(snapshot):
    for p, s in zip(model_parameters, snapshot):
        p.set_value(s, borrow=False)


# In[58]:

# init training

i=0
e=0

init_parameters()
for v in velocities:
    v.set_value(np.zeros_like(v.get_value()))

best_valid_error_rate = np.inf
best_params = snapshot_parameters()
best_params_epoch = 0

train_erros = []
train_loss = []
train_nll = []
validation_errors = []

number_of_epochs = 3
patience_expansion = 1.5


if len(sys.argv) == 3:
	print "Loading saved parameters"
	f = open(sys.argv[2], 'rb')
	saved_snapshot = cPickle.load(f)
	f.close()
	load_parameters(saved_snapshot)


print "Starting training"

# In[62]:

# training loop
try:
    while e<number_of_epochs: #This loop goes over epochs
        e += 1
        #First train on all data from this batch
        epoch_start_i = i
        for X_batch, Y_batch in cifar_train_stream.get_epoch_iterator(): 
            i += 1

            K = 1000
            lrate = 4e-3 * K / np.maximum(K, i)
            momentum=0.9

            L, err_rate, nll, wdec = train_step(X_batch, Y_batch, lrate, momentum)

            #print [p.get_value().ravel()[:10] for p in model_parameters]
            #print [p.get_value().ravel()[:10] for p in velocities]


            train_loss.append((i,L))
            train_erros.append((i,err_rate))
            train_nll.append((i,nll))
            if i % 100 == 0:
                print "At minibatch %d, batch loss %f, batch nll %f, batch error rate %f%%" % (i, L, nll, err_rate*100)

        # After an epoch compute validation error
        val_error_rate = compute_error_rate(cifar_validation_stream)
        if val_error_rate < best_valid_error_rate:
            number_of_epochs = np.maximum(number_of_epochs, e * patience_expansion+1)
            best_valid_error_rate = val_error_rate
            best_params = snapshot_parameters()
            best_params_epoch = e
        validation_errors.append((i,val_error_rate))
        print "After epoch %d: valid_err_rate: %f%% currently going to do %d epochs" %(
            e, val_error_rate*100, number_of_epochs)
        print "After epoch %d: averaged train_err_rate: %f%% averaged train nll: %f averaged train loss: %f" %(
            e, np.mean(np.asarray(train_erros)[epoch_start_i:,1])*100, 
            np.mean(np.asarray(train_nll)[epoch_start_i:,1]),
            np.mean(np.asarray(train_loss)[epoch_start_i:,1]))
except KeyboardInterrupt:
    pass
finally:
    print "Setting network parameters from after epoch %d" %(best_params_epoch)
    load_parameters(best_params)
    err_rate_z = compute_error_rate(cifar_test_stream)*100.0
    print "Test error rate is %f%%" %(err_rate_z,)
    f = open(sys.argv[1] + ' params ' + '%.2f percent'%(err_rate_z,),'wb')
    cPickle.dump(best_params,f)
    f.close()
    print "Parameters saved"

