#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov  1 06:43:06 2021

@author: natrask
"""

import numpy as np
# import tensorflow as tf
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()                                                                                                                                                        
sess = tf.Session()
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.datasets import mnist
# import tensorflow_probability as tfp
import matplotlib.pyplot as plt
import tqdm
from scipy import stats
#all parameters up top
clusters = 10
encoding_dim = 2
digits = 10
pouwidth=10
lrate = 0.5e-4          
inputShape = 784 #28^2 pixels in MNIST
Ndata = 20
noiseLevel = 0.05
npix1d = 28
minibatch_size = 100
stab = 1e-15
# The usual way to lost mnist
#(train_X, train_y), (test_X, test_y) = mnist.load_data()

# If the SNL proxy is blocking this though you'll need to download it yourself
# (copied and pasted from stack overflow)

# "You do not need additional code for that but can tell load_data to load a local version in the first place:

# You can download the file https://s3.amazonaws.com/img-datasets/mnist.npz from another computer with proper (proxy) access (taken from https://github.com/keras-team/keras/blob/master/keras/datasets/mnist.py),
# copy it to the the directory ~/.keras/datasets/ (on Linux and macOS)
# and run load_data(path='mnist.npz') with the right file name"

(mnistX_train, mnisty_train), (mnistX_test, mnisty_test) = mnist.load_data()
train_mask = np.isin(mnisty_train, np.array(range(digits))) #just grab a couple mnist digits
mnistX_train = mnistX_train[train_mask]
mnisty_train = mnisty_train[train_mask]

x = np.linspace(0,1,Ndata)
y_train = np.einsum('i,j->ij',x,mnisty_train) + np.random.normal(0,noiseLevel,(Ndata,mnisty_train.shape[0]))
y_test = np.einsum('i,j->ij',x,mnisty_test) + np.random.normal(0,noiseLevel,(Ndata,mnisty_test.shape[0]))

#for postprocessing
sortedmean=[np.mean(y_test.T[np.where(mnisty_test==z)].T,axis=1) for z in range(digits)]
sortedstd=[np.std(y_test.T[np.where(mnisty_test==z)].T,axis=1) for z in range(digits)]

mnistX_train = mnistX_train.astype('float32') / 255.
mnistX_test = mnistX_test.astype('float32') / 255.
Nbatches = mnistX_train.shape[0]

# convert class vectors to binary class matrices
yc_train = keras.utils.to_categorical(mnisty_train, digits)
yc_test = keras.utils.to_categorical(mnisty_train, digits)

# Want to do unsupervised learning:
#     From input pixels (mnistX) and output series, infer latent variables Z
#plt.plot(x,y_test) # to look at what we're attempting to learn labels from (expensive so keep commented)

#Some pdfs that we'll need
def log_GMM_pdf(sample, mix,mean, logvar, raxis=1):
    piInv2 = tf.pow(tf.cast((2. * np.pi),dtype=tf.float32),-encoding_dim/2)
    SigmaInv = tf.exp(-logvar)
    zshift = tf.expand_dims(sample,1)-mean
    expo = tf.exp(-0.5*tf.einsum('bpz,pz,bpz->bp',zshift,SigmaInv,zshift))
    detsig = tf.pow(tf.reduce_prod(SigmaInv,axis=1),0.5)
    logsamplepdf = tf.log(piInv2*tf.einsum('p,p,bp->b',mix,detsig,expo))
    return logsamplepdf
def log_normal_pdf(sample, mean, logvar, raxis=1):
  log2pi = tf.cast(tf.math.log(2. * np.pi),dtype=tf.float32)
  return tf.reduce_sum(
      -.5 * ((sample - mean) ** 2. *tf.exp(-logvar) + logvar + log2pi),
      axis=raxis)

#Encoder: input = image, output = mu_q, logvar_q to parameterization gaussian
notEncoderVars = tf.all_variables()
input_shape = (28, 28, 1)
encoder = keras.Sequential(
    [
        keras.Input(shape=input_shape,dtype=tf.float32),
        layers.Conv2D(32, kernel_size=(3, 3), activation="elu",dtype=tf.float32),
        tf.keras.layers.BatchNormalization(dtype=tf.float32),
        layers.Conv2D(64, kernel_size=(3, 3), activation="elu",dtype=tf.float32),
        tf.keras.layers.BatchNormalization(dtype=tf.float32),
        layers.Flatten(),
        layers.Dense(encoding_dim + encoding_dim),
    ]
)
encoder.summary()


encoder2 = keras.Sequential()
encoder2.add(keras.Input(shape=(Ndata,1),dtype=tf.float32))
encoder2.add(tf.keras.layers.Conv1D(8, kernel_size=(3), activation="elu",dtype=tf.float32))
encoder2.add(tf.keras.layers.BatchNormalization(dtype=tf.float32),)
encoder2.add(tf.keras.layers.Conv1D(16, kernel_size=(3), activation="elu",dtype=tf.float32))
encoder2.add(tf.keras.layers.BatchNormalization(dtype=tf.float32),)
encoder2.add(tf.keras.layers.Flatten())
encoder2.add(tf.keras.layers.Dense(encoding_dim + encoding_dim))
encoder2.summary()


input_img = tf.placeholder(dtype=tf.float32,shape=(None,npix1d,npix1d))
inputlabels = tf.placeholder(dtype=tf.float32,shape=(None,Ndata))
mu_q1, logvar_q1 = tf.split(encoder(tf.expand_dims(input_img,-1)), num_or_size_splits=2, axis=1)
mu_q2, logvar_q2 = tf.split(encoder2(tf.expand_dims(inputlabels,-1)), num_or_size_splits=2, axis=1)

#Either take both modalities to produce q(z|x1,x2):
# merge = tf.concat([encoder(tf.expand_dims(input_img,-1)),encoder2(tf.expand_dims(inputlabels,-1))],1)
# densemerge  = tf.keras.layers.Dense(2*encoding_dim)(merge)
# mu_q, logvar_q = tf.split(densemerge, num_or_size_splits=2, axis=1)

#or try to put together q(z|x1,x2)=q(z|x1)*q(z|x2)
logvar_q = -tf.log(tf.exp(-logvar_q1)+tf.exp(-logvar_q2))
mu_q = tf.exp(logvar_q)*(tf.exp(-logvar_q1)*mu_q1+tf.exp(-logvar_q2)*mu_q2)

#Don't think we actually need to scale qz_x: the one here does not integrate to 1!
# scale_mu = tf.exp(logvar_q1-logvar_q2)*mu_q2
# scale_logvar = 2*logvar_q1-logvar_q
# Zc = tf.exp(log_normal_pdf(mu_q1, scale_mu, scale_logvar))

#reparameterization tricks
Z_eps = tf.random.normal(shape=tf.shape(mu_q),dtype=tf.float32) #noise 
Z_eps1 = tf.random.normal(shape=tf.shape(mu_q),dtype=tf.float32) #noise 
Z_eps2 = tf.random.normal(shape=tf.shape(mu_q),dtype=tf.float32) #noise 
Z  = (Z_eps *tf.exp(logvar_q * .5) + mu_q )
Z1 = (Z_eps1 *tf.exp(logvar_q1 * .5) + mu_q1 )
Z2 = (Z_eps2 *tf.exp(logvar_q2 * .5) + mu_q2 )
# Z = tf.einsum('bz,b->bz',Z,Zc) #the alleged proper scaling?
encoderVars =  list(set(tf.all_variables()) - set(notEncoderVars))

# mu_qhat = t

#decoder for sampling from latent distribution
notDecoderVars = tf.all_variables()
decoder = keras.Sequential(
    [
        keras.Input(shape=encoding_dim,dtype=tf.float32),
        layers.Dense(units=7*7*32, activation=tf.nn.leaky_relu),
        layers.Reshape(target_shape=(7, 7, 32)),
        layers.Conv2DTranspose(
            filters=64, kernel_size=3, strides=2, padding='same',
            activation='relu'),
        layers.Conv2DTranspose(
            filters=32, kernel_size=3, strides=2, padding='same',
            activation='relu'),
        layers.Conv2DTranspose(
            filters=1, kernel_size=3, strides=1, padding='same'),
    ]
)
mu_x1 = decoder(Z)
logvar_x1 = tf.zeros(tf.shape(mu_x1),dtype=tf.float32)#hardcode unit variance output
decoderVars =  list(set(tf.all_variables()) - set(notDecoderVars))
logpx1_z = log_normal_pdf(tf.expand_dims(input_img,-1), mu_x1, logvar_x1)  


#Prior parameters
pi = tf.nn.softmax(tf.Variable((1./clusters)*np.ones(clusters),dtype=tf.float32))
hpack = np.power(clusters,-1/encoding_dim)
mixmean = tf.Variable(1e-3*np.random.normal(size=(clusters,encoding_dim)),dtype=tf.float32)
mixlogvar = tf.Variable(np.log(hpack**2)*np.ones((clusters,encoding_dim)),dtype=tf.float32) #hardcode ball packing lengthscale?
# mixlogvar = tf.Variable(np.zeros((clusters,encoding_dim)),dtype=tf.float32) #hardcode ball packing lengthscale?
logpz_c = log_GMM_pdf(Z,pi,mixmean,mixlogvar) #gaussian mixture posterior, initialized to unit gaussian

#Expert models - just learning the slope x2 = theta*t + noise
t = tf.constant(x,dtype=tf.float32)
theta_init = np.mean(np.einsum('i,ib->b',x,y_train)/np.sum(x*x))
theta = tf.Variable(np.random.normal(theta_init,0.01,clusters),dtype=tf.float32)
mu_x2 = t*tf.expand_dims(theta,-1)
logvar_x2 = tf.expand_dims(tf.Variable(np.zeros(clusters),dtype=tf.float32),0)


#posterior gamma = p(c|x) evaluated via bayes rule
px1_c_a = tf.reduce_prod(tf.exp(-0.5*tf.pow(tf.expand_dims(Z1,1)-mixmean,2)/tf.exp(mixlogvar)),-1)
px1_c_b = tf.reduce_prod(tf.exp(-0.5*mixlogvar),axis=1)
px2_c_a = tf.reduce_prod(tf.exp(-0.5*tf.pow(tf.expand_dims(inputlabels,1)-tf.expand_dims(mu_x2,0),2)*tf.expand_dims(tf.exp(-logvar_x2),-1)),-1)
px2_c_b = tf.transpose(tf.exp(-0.5*logvar_x2))
# px12_c = tf.einsum('bp,p,bp,pz->bp',px1_c_a,px1_c_b,px2_c_a,px2_c_b)
px12_c = tf.einsum('bp,p->bp',px1_c_a,px1_c_b)
# px12_c = tf.einsum('bp,pz->bp',px2_c_a,px2_c_b)

gammanum = tf.einsum('p,bp->bp',pi,px12_c)+1e-10


# post1 = tf.reduce_prod(tf.exp(-0.5*tf.pow(tf.expand_dims(Z,1)-mixmean,2)/tf.exp(mixlogvar)),-1)
# invdetsig = tf.reduce_prod(tf.exp(-0.5*mixlogvar),axis=1)
# gammanum = tf.einsum('p,p,bp->bp',pi,invdetsig,post1)+1e-6
gamma = gammanum/tf.expand_dims(tf.reduce_sum(gammanum,1),-1)

post2 = tf.reduce_prod(tf.exp(-0.5*tf.pow(tf.expand_dims(Z1,1)-mixmean,2)/tf.exp(mixlogvar)),-1)
invdetsig2 = tf.reduce_prod(tf.exp(-0.5*mixlogvar),axis=1)
gammanum2 = tf.einsum('p,p,bp->bp',pi,invdetsig2,post2)+1e-6
gamma2 = gammanum2/tf.expand_dims(tf.reduce_sum(gammanum2,1),-1)

#for predicting x2,x1 from unimodal data
predx2 = tf.einsum('bp,px->bx',gamma,mu_x2)
predx2_unimodal = tf.einsum('bp,px->bx',gamma2,mu_x2)
predx1_unimodal = decoder(Z1)


#Build losses
#Reconstruction loss for X1 and X2
L1 = tf.reduce_sum(tf.pow(tf.expand_dims(input_img,-1)-mu_x1,2)) 
Qa = 0.5*tf.reduce_sum(tf.pow(tf.expand_dims(inputlabels,1)-tf.expand_dims(mu_x2,0),2)*tf.expand_dims(tf.exp(-logvar_x2),-1),-1)
Qb = Ndata*logvar_x2
LOSSexpert = tf.reduce_sum(tf.einsum('bp,bp->b',Qa+Qb,gamma))

#KL divergence terms
L2a = mixlogvar+tf.expand_dims(tf.exp(logvar_q),1)
L2b = tf.expand_dims(tf.exp(logvar_q),1)/tf.expand_dims(tf.exp(mixlogvar),0)
L2c = tf.pow(tf.expand_dims(mixmean,0)-tf.expand_dims(mu_q,1),2)/tf.exp(mixlogvar)
L2  = 0.5*tf.reduce_sum(tf.einsum('bc,bc->b',gamma,tf.reduce_sum(L2a+L2b+L2c,axis=2)))
L3 = -tf.reduce_sum(tf.einsum('bp,bp->b',gamma,tf.log(tf.expand_dims(pi,0)/gamma)))
L4 = -0.5*tf.reduce_sum(tf.einsum('bz->bz',logvar_q))





# Code for assigning prior mixtures with expectation maximization
newmixmean = tf.placeholder(shape=(clusters,encoding_dim),dtype=tf.float32)
updatemeans = mixmean.assign(newmixmean)
newmixvars = tf.placeholder(shape=(clusters,encoding_dim),dtype=tf.float32)
updatevars = mixlogvar.assign(tf.log(newmixvars))
newtheta = tf.placeholder(shape=(clusters,),dtype=tf.float32)


#First term is reconstruction, others are KL divergence penalties
LOSS = (L1 + LOSSexpert) + (L2 + L3 + L4) 

#Set up optimizers
VAEoptimizer = tf.train.AdamOptimizer(learning_rate=lrate).minimize(LOSS)
sess.run(tf.global_variables_initializer())   
graph = tf.get_default_graph()

#to postprocess centers of distributions
decodemeans = decoder(mixmean)


x1erroruni = tf.sqrt(tf.reduce_mean((predx1_unimodal-tf.expand_dims(input_img,-1))**2)/tf.reduce_mean((inputlabels)**2))
x2error = tf.sqrt(tf.reduce_mean((predx2-inputlabels)**2)/tf.reduce_mean((inputlabels)**2))
x2erroruni = tf.sqrt(tf.reduce_mean((predx2_unimodal-inputlabels)**2)/tf.reduce_mean((inputlabels)**2))
#Train
losslog=[]
losslog2=[]
losslog3=[]
losslog4=[]
frame = 0
for its in range(1000):
    #viz before each epoch
    nviz = 1000
    labels = mnisty_train[:nviz]
    batchX = mnistX_train[:nviz,:,:]
    batchY = y_train[:,:nviz].T
    print(sess.run(LOSS,feed_dict={input_img:batchX,inputlabels:batchY}))
    Zscat = sess.run(mu_q,feed_dict={input_img:batchX,inputlabels:batchY})
    npdecoded = sess.run(decodemeans)
    fig = plt.figure(constrained_layout=True,figsize=[25,15])
    subplots = fig.subfigures(3 ,5)
    zax1 = subplots[0][0].subplots(1,1)
    zax1.scatter(Zscat[:,0],Zscat[:,1],c=labels)
    centers = sess.run(mixmean)
    zax1.scatter(centers[:,0],centers[:,1],c='red',s=150)
    zax2 = subplots[0][1].subplots(1,1)
    zax2.plot(x,batchY[:nviz,:].T,'.',c='black')
    zax2.plot(x,sess.run(mu_x2).T)
    zax3 = subplots[0][2].subplots(1,1)
    zax3.plot(losslog)
    zax4 = subplots[0][3].subplots(1,1)
    zax4.plot(losslog2)
    zax4.plot(losslog3)
    zax4.plot(losslog4)
    for z in range(5):
        zax = subplots[1][z].subplots(1,1)
        zax.imshow(npdecoded[z,:,:,0], cmap=plt.get_cmap('gray'))
    for z in range(5,10):
        zax = subplots[2][z-5].subplots(1,1)
        zax.imshow(npdecoded[z,:,:,0], cmap=plt.get_cmap('gray'))
    fig.savefig("MNIST-EM{:03d}.png".format(frame))
    frame = frame + 1
    
    wsum = np.zeros((clusters),dtype=np.float32)
    mean = np.zeros((clusters,encoding_dim),dtype=np.float32)
    S = np.zeros((clusters,encoding_dim),dtype=np.float32)
    
    #Run Epoch
    numbatches = int(len(mnistX_train) / minibatch_size)
    shufflin = np.random.permutation(np.arange(numbatches))
    for jj in tqdm.tqdm(range(numbatches)):
        j = shufflin[jj]
        batchX = mnistX_train[j*minibatch_size:(j+1)*minibatch_size,:,:]
        batchY = y_train[:,j*minibatch_size:(j+1)*minibatch_size].T
        
        #EM step if we're going to do it. 
        #get zs and posterior weights to update GMM vars
        delgamma = sess.run(gamma,feed_dict={input_img:batchX,inputlabels:batchY})
        delz = sess.run(Z,feed_dict={input_img:batchX,inputlabels:batchY})
        delmu = sess.run(mu_q,feed_dict={input_img:batchX,inputlabels:batchY})

        wsumold = wsum
        wsum = wsum + delgamma.sum(0)
        #mean computation
        mean_old = mean
        mean = (np.expand_dims(wsumold,-1)*mean_old + (np.expand_dims(delgamma,-1)*np.expand_dims(delmu,1)).sum(0))/np.expand_dims(wsum,1)
        #variance computation
        # S_old = S
        # S = (np.expand_dims(wsumold,-1)*S_old + (np.expand_dims(delgamma,-1)*(np.expand_dims(delz,1)-mean)**2).sum(0))/np.expand_dims(wsum,1)
        #update model
        sess.run(updatemeans,feed_dict={newmixmean:mean})
        # sess.run(updatevars,feed_dict={newmixvars:S})
  
    for jj in tqdm.tqdm(range(numbatches)):
        j = shufflin[jj]
        batchX = mnistX_train[j*minibatch_size:(j+1)*minibatch_size,:,:]
        batchY = y_train[:,j*minibatch_size:(j+1)*minibatch_size].T
        sess.run(VAEoptimizer,feed_dict={input_img:batchX,inputlabels:batchY})
    
    #postprocess accuracy based off mode of labels
    batchX = mnistX_test[:,:,:]
    batchY = y_test[:,:].T
    predclass = np.argmax(sess.run(gamma,feed_dict={input_img:batchX,inputlabels:batchY}),axis=1)
    classlabel = [stats.mode(predclass[np.isin(mnisty_test, [z])])[0][0] for z in range(clusters)]
    totalcorrect = [np.sum(predclass[np.isin(mnisty_test, [z])]==classlabel[z]) for z in range(clusters)]
    totaltested=np.isin(mnisty_test, [0,1,2,3,4,5,6,7,8,9]).sum()
    loss = np.sum(totalcorrect)/totaltested
    losslog.append(loss)
    loss2 = sess.run(x2error,feed_dict={input_img:batchX,inputlabels:batchY})
    losslog2.append(loss2)
    
    loss3 = sess.run(x1erroruni,feed_dict={input_img:batchX,inputlabels:batchY})
    loss4 = sess.run(x2erroruni,feed_dict={input_img:batchX,inputlabels:batchY})
    losslog3.append(loss3)
    losslog4.append(loss4)

    print(str(its)+' accuracy on test: ' + str(loss) + ' ' + str(loss2))