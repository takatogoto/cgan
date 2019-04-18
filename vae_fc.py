import numpy as np
import tensorflow as tf
import math
import os, time, itertools, pickle, random, glob, imageio
from PIL import Image
from tensorflow.examples.tutorials.mnist import input_data
from tensorflow.contrib.slim import fully_connected as fc
import matplotlib.pyplot as plt

# Load albumdata

img_size = 64
num_epoch =1000
batch_size = 64
n_z = 64
z_val_name = 'z64_tras_ep1000_noinvers.npy'
z_list_name = 'z64_list_ep1000_noinvers.npy'


samples = np.load('vae/x_labels_64.npy')
#samples = (1 - samples)
#samples = np.load('vae/x_samples_64.npy')
#labels = np.load('vae/x_labels_64.npy')

input_dim = img_size * img_size * 3
num_sample = samples.shape[0]

class VariantionalAutoencoder(object):

    def __init__(self, input_dim =input_dim, learning_rate=1e-4, batch_size=64, n_z=16):
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.n_z = n_z
        self.input_dim = input_dim

        tf.reset_default_graph()
        self.build()

        self.sess = tf.InteractiveSession(config=tf.ConfigProto(
        allow_soft_placement=True,log_device_placement=True))
        self.sess.run(tf.global_variables_initializer())


    # Build the netowrk and the loss functions
    def build(self):
        
        self.x = tf.placeholder(
            name='x', dtype=tf.float32, shape=[None, self.input_dim])

        # Encode
        # x -> z_mean, z_sigma -> z
        f1 = fc(self.x, 2048, scope='enc_fc1', activation_fn=tf.nn.relu)
        print(f1)
        f2 = fc(f1, 1024, scope='enc_fc2', activation_fn=tf.nn.relu)
        f3 = fc(f2, 512, scope='enc_fc3', activation_fn=tf.nn.relu)
        f4 = fc(f3, 256, scope='enc_fc4', activation_fn=tf.nn.relu)
        f5 = fc(f4, 128, scope='enc_fc5', activation_fn=tf.nn.relu)
        f6 = fc(f5, 64, scope='enc_fc6', activation_fn=tf.nn.relu)
        self.z_mu = fc(f6, self.n_z, scope='enc_fc7_mu', 
                       activation_fn=None)
        self.z_log_sigma_sq = fc(f6, self.n_z, scope='enc_fc7_sigma', 
                                 activation_fn=None)
        eps = tf.random_normal(
            shape=tf.shape(self.z_log_sigma_sq),
            mean=0, stddev=1, dtype=tf.float32)
        self.z = self.z_mu + tf.sqrt(tf.exp(self.z_log_sigma_sq)) * eps

        # Decode
        # z -> x_hat
        g1 = fc(self.z, 64, scope='dec_fc1', activation_fn=tf.nn.relu)
        g2 = fc(g1, 128, scope='dec_fc2', activation_fn=tf.nn.relu)
        g3 = fc(g2, 256, scope='dec_fc3', activation_fn=tf.nn.relu)
        g4 = fc(g3, 1024, scope='dec_fc4', activation_fn=tf.nn.relu)
        g5 = fc(g4, 1024, scope='dec_fc5', activation_fn=tf.nn.relu)
        g6 = fc(g5, 2048, scope='dec_fc6', activation_fn=tf.nn.relu)
        self.x_hat = fc(g6, self.input_dim, scope='dec_fc7', 
                        activation_fn=tf.sigmoid)

        # Loss
        # Reconstruction loss
        # Minimize the cross-entropy loss
        # H(x, x_hat) = -\Sigma x*log(x_hat) + (1-x)*log(1-x_hat)
        epsilon = 1e-10
        recon_loss = -tf.reduce_sum(
            self.x * tf.log(epsilon+self.x_hat) + 
            (1-self.x) * tf.log(epsilon+1-self.x_hat), 
            axis=1
        )
        self.recon_loss = tf.reduce_mean(recon_loss)

        # Latent loss
        # KL divergence: measure the difference between two distributions
        # Here we measure the divergence between 
        # the latent distribution and N(0, 1)
        latent_loss = -0.5 * tf.reduce_sum(
            1 + self.z_log_sigma_sq - tf.square(self.z_mu) - 
            tf.exp(self.z_log_sigma_sq), axis=1)
        self.latent_loss = tf.reduce_mean(latent_loss)

        self.total_loss = self.recon_loss + self.latent_loss
        self.train_op = tf.train.AdamOptimizer(
            learning_rate=self.learning_rate).minimize(self.total_loss)
        
        self.losses = {
            'recon_loss': self.recon_loss,
            'latent_loss': self.latent_loss,
            'total_loss': self.total_loss,
        }        
        return

    # Execute the forward and the backward pass
    def run_single_step(self, x):
        _, losses = self.sess.run(
            [self.train_op, self.losses],
            feed_dict={self.x: x}
        )
        return losses

    # x -> x_hat
    def reconstructor(self, x):
        x_hat = self.sess.run(self.x_hat, feed_dict={self.x: x})
        return x_hat

    # z -> x
    def generator(self, z):
        x_hat = self.sess.run(self.x_hat, feed_dict={self.z: z})
        return x_hat
    
    # x -> z
    def transformer(self, x):
        z = self.sess.run(self.z, feed_dict={self.x: x})
        return z
    
def trainer_album(model_object, sample, input_dim =input_dim, learning_rate=1e-4, 
            batch_size=16, num_epoch=5, n_z=16, log_step=5,
                 num_sample = num_sample):
    model = model_object(
        learning_rate=learning_rate, batch_size=batch_size, n_z=n_z,
    input_dim =input_dim)
    
    
    step = 0

    for epoch in range(num_epoch):
        start_time = time.time()
        for iter in range(num_sample // batch_size):
            step += 1
            # Get a batch
            batch = sample[iter * batch_size : (iter + 1) * batch_size]
            # Execute the forward and backward pass 
            # Report computed losses
            #print('batch',batch)
            losses = model.run_single_step(batch)
        end_time = time.time()
        
        if epoch % log_step == 0:
            log_str = '[Epoch {}] '.format(epoch)
            for k, v in losses.items():
                log_str += '{}: {:.3f}  '.format(k, v)
            log_str += '({:.3f} sec/epoch)'.format(end_time - start_time)
            print(log_str)
            
    print('Done!')
    return model

tf.reset_default_graph()

with tf.device('/gpu:0'):
    model_2d_vae = trainer_album(VariantionalAutoencoder, samples.reshape(-1,input_dim), 
                             num_epoch=num_epoch, batch_size=batch_size, n_z=n_z, 
                             input_dim =input_dim, num_sample=num_sample)
    
# get and save latent z
z_transform = model_2d_vae.transformer(samples.reshape(-1,img_size*img_size*3))
np.save(z_val_name, z_transform)

plt.imshow(samples[100])
plt.savefig('a_xs_ep1000_noinvers.png')

x_hat1 = model_2d_vae.reconstructor(samples[100].reshape(-1, 64*64*3))
plt.imshow(x_hat1.reshape(64,64,3))
plt.savefig('a_xhats_ep1000_noinvers.png') 
