import numpy as np
import theano
import theano.tensor as T
import lasagne


import skimage.transform
import sklearn.cross_validation
import pickle
import os



##build the vgg model

from lasagne.layers import InputLayer, DenseLayer, NonlinearityLayer
from lasagne.layers.dnn import Conv2DDNNLayer as ConvLayer
from lasagne.layers import Pool2DLayer as PoolLayer
from lasagne.nonlinearities import softmax
from lasagne.utils import floatX

def build_model():
    net = {}
    net['input'] = InputLayer((None, 3, 224, 224))
    net['conv1_1'] = ConvLayer(net['input'], 64, 3, pad=1,trainable=False)
    net['conv1_2'] = ConvLayer(net['conv1_1'], 64, 3, pad=1,trainable=False)
    net['pool1'] = PoolLayer(net['conv1_2'], 2)
    net['conv2_1'] = ConvLayer(net['pool1'], 128, 3, pad=1,trainable=False)
    net['conv2_2'] = ConvLayer(net['conv2_1'], 128, 3, pad=1,trainable=False)
    net['pool2'] = PoolLayer(net['conv2_2'], 2)
    net['conv3_1'] = ConvLayer(net['pool2'], 256, 3, pad=1,trainable=False)
    net['conv3_2'] = ConvLayer(net['conv3_1'], 256, 3, pad=1,trainable=False)
    net['conv3_3'] = ConvLayer(net['conv3_2'], 256, 3, pad=1,trainable=False)
    net['pool3'] = PoolLayer(net['conv3_3'], 2)
    net['conv4_1'] = ConvLayer(net['pool3'], 512, 3, pad=1,trainable=False)
    net['conv4_2'] = ConvLayer(net['conv4_1'], 512, 3, pad=1,trainable=False)
    net['conv4_3'] = ConvLayer(net['conv4_2'], 512, 3, pad=1,trainable=False)
    net['pool4'] = PoolLayer(net['conv4_3'], 2)
    net['conv5_1'] = ConvLayer(net['pool4'], 512, 3, pad=1,trainable=False)
    net['conv5_2'] = ConvLayer(net['conv5_1'], 512, 3, pad=1)
    net['conv5_3'] = ConvLayer(net['conv5_2'], 512, 3, pad=1)

    net['pool5'] = PoolLayer(net['conv5_3'], 2)
    net['fc6'] = DenseLayer(net['pool5'], num_units=4096)
    net['fc7'] = DenseLayer(net['fc6'], num_units=4096)
    net['fc8'] = DenseLayer(net['fc7'], num_units=1000, nonlinearity=None)
    net['prob'] = NonlinearityLayer(net['fc8'], softmax)

    return net




# Load model weights and metadata
d = pickle.load(open('vgg16.pkl'))

#setting batch size for the whole training
BATCH_SIZE = 50
MEAN_IMAGE=np.load('vgg_mean.npy')


# Build the network and fill with pretrained weights
net = build_model()

lasagne.layers.set_all_param_values(net['prob'], d['param values'])
# lasagne.layers.set_all_param_values(net['prob'], d['param values'][22:32],trainable=True)


print 'successfully...'






# import h5py
# f=h5py.File('test.h5','r')
# ##get the dataset
# tr_data=f['tr_data']
# tr_lb=f['tr_lb']
# ts_data=f['ts_data']
# ts_lb=f['ts_lb']

# MEAN_IMAGE=np.load('vgg_mean.npy')
# #error occur they need to be dtype float32
# #error occur they need to be dtype float32
# tr_data=tr_data[:,:,16:240,16:240].astype('float32')
# ts_data=ts_data[:,:,16:240,16:240].astype('float32')
# tr_data=tr_data-MEAN_IMAGE
# ts_data=ts_data-MEAN_IMAGE
# tr_lb=tr_lb[:].astype('int32')
# ts_lb=ts_lb[:].astype('int32')
# print tr_data.shape

##begin to train the model 

##first just configure
half_feature_layer=DenseLayer(net['fc6'],num_units=2048)
output_layer=DenseLayer(half_feature_layer,num_units=7,nonlinearity=softmax)
final_prob=NonlinearityLayer(output_layer, softmax)


with np.load('model_vgg16_CIFE_150.npz') as f:
    param_values = [f['arr_%d' % i] for i in range(len(f.files))]

lasagne.layers.set_all_param_values(output_layer, param_values)


# Define loss function and metrics, and get an updates dictionary
X_sym = T.tensor4()
y_sym = T.ivector()

prediction = lasagne.layers.get_output(final_prob, X_sym)
loss = lasagne.objectives.categorical_crossentropy(prediction, y_sym)
loss = loss.mean()

acc = T.mean(T.eq(T.argmax(prediction, axis=1), y_sym),
                      dtype=theano.config.floatX)

params = lasagne.layers.get_all_params(final_prob, trainable=True)
updates = lasagne.updates.nesterov_momentum(
        loss, params, learning_rate=0.0001, momentum=0.9)


# Compile functions for training, validation and prediction
train_fn = theano.function([X_sym, y_sym], loss, updates=updates)
val_fn = theano.function([X_sym, y_sym], [loss, acc])
pred_fn = theano.function([X_sym], prediction)



# generator splitting an iterable into chunks of maximum length N
def batches(iterable, N):
    chunk = []
    for item in iterable:
        chunk.append(item)
        if len(chunk)==N:
            rst=chunk
            chunk=[]
            yield rst
    if chunk:
        yield chunk

# We need a fairly small batch size to fit a large network like this in GPU memory



def train_batch():
    trdata,trlb=imdata(imglist)
    # trdata=trdata-MEAN_IMAGE
    return train_fn(trdata,trlb)

def val_batch():
    ix = range(len(y_val))
    np.random.shuffle(ix)
    ix = ix[:BATCH_SIZE]
    return val_fn(X_val[ix], y_val[ix])


###read the emotion dataset make them as numpy array


#load the files first then deliver thedata when needed.

import random
import cv2
def imdata(fls,batchsize=BATCH_SIZE):
    datablob=np.ndarray((batchsize,3,224,224))
    datalb=np.zeros((batchsize,))
    n=len(fls)
    random.shuffle(fls)
    fls=fls[:batchsize]
    im224=np.zeros((3,224,224))
    for i,f in enumerate(fls):
        fname,flabel=f.split(' ')
        
        imi=cv2.imread(fname)
        #cv2 read img as 3xNxN and with BGR
        if imi==None:continue
        for t in range(3):
            im224[t,:,:]=cv2.resize(imi[:,:,t],(224,224))
        datablob[i,:,:,:]=im224

        #then the label 
        datalb[i]=int(flabel)
    datablob=datablob.astype('float32')
    datalb=datalb.astype('int32')
    return datablob,datalb
def testbatch(fls):
    Num=len(fls)
    datablob=np.ndarray((Num,3,224,224))
    datalb=np.zeros((Num,))
    im224=np.zeros((3,224,224))
    for i,f in enumerate(fls):
        fname,flabel=f.split(' ')

        imi=cv2.imread(fname)
        #cv2 read img as 3xNxN and with BGR
        if imi==None:
            # print fname
            fname,flabel=fls[0].split(' ')
            imi=cv2.imread(fname)
        #cv2 read img as 3xNxN and with BGR
        if imi==None:continue
        for t in range(3):
            im224[t,:,:]=cv2.resize(imi[:,:,t],(224,224))
        datablob[i,:,:,:]=im224

        #then the label 
        datalb[i]=int(flabel)
    datablob=datablob.astype('float32')
    # datablob-=MEAN_IMAGE
    datalb=datalb.astype('int32')
    return datablob,datalb



listtrainpath='train_emo.txt'
listtestpath='test_emo.txt'
impath='/home/wei/caffedata/webim/webemo_trag'
testpath='/home/wei/caffedata/webim/CIFEv2.0/webemo_ts'

fp=open(listtrainpath)
imglist=fp.readlines()

#reading test list,ixx contain all the test image names
ft=open(listtestpath)
ixx=ft.readlines()

confu_mt=np.zeros((7,7))

cnt=0
for i,l in enumerate(ixx):

    fname,flabel=l[:-1].split(' ')
    ft_data,ft_lb=imdata([l],1)

    prob=pred_fn(ft_data)
    # if i<10:print fname,ft_feat
    prob_i=np.argmax(prob)
    confu_mt[int(ft_lb),prob_i]+=1
    if prob_i==int(ft_lb):cnt+=1

for i in range(7):
    confu_mt[i,:]/=np.sum(confu_mt[i,:])
print confu_mt

print 'acc: ', float(cnt)/len(ixx)


#without clearance
'''[[ 0.81611208  0.01401051  0.02101576  0.03152364  0.04903678  0.04028021
   0.02802102]
 [ 0.1369863   0.23287671  0.02054795  0.06164384  0.35616438  0.11986301
   0.07191781]
 [ 0.06763285  0.02898551  0.62801932  0.0410628   0.05555556  0.03381643
   0.14492754]
 [ 0.02385321  0.01376147  0.00458716  0.89633028  0.01559633  0.02844037
   0.01743119]
 [ 0.04761905  0.06442577  0.00840336  0.02521008  0.77170868  0.04341737
   0.03921569]
 [ 0.07248322  0.01879195  0.0147651   0.03624161  0.02818792  0.82416107
   0.00536913]
 [ 0.02173913  0.02675585  0.09364548  0.02842809  0.07023411  0.00501672
   0.7541806 ]]
acc:  0.765596745027
'''

#with clearance

'''[[ 0.81611208  0.01401051  0.02101576  0.03152364  0.04903678  0.04028021
   0.02802102]
 [ 0.1369863   0.23287671  0.02054795  0.06164384  0.35616438  0.11986301
   0.07191781]
 [ 0.06763285  0.02898551  0.62801932  0.0410628   0.05555556  0.03381643
   0.14492754]
 [ 0.02385321  0.01376147  0.00458716  0.89633028  0.01559633  0.02844037
   0.01743119]
 [ 0.04761905  0.06442577  0.00840336  0.02521008  0.77170868  0.04341737
   0.03921569]
 [ 0.07248322  0.01879195  0.0147651   0.03624161  0.02818792  0.82416107
   0.00536913]
 [ 0.02173913  0.02675585  0.09364548  0.02842809  0.07023411  0.00501672
   0.7541806 ]]
acc:  0.765596745027
'''

