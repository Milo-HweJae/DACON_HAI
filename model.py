# -*- coding: utf-8 -*-
"""
Created on Mon May 11 15:05:13 2020
@author: gnos
"""

import tensorflow as tf
from tensorflow.keras.layers import Dense, Flatten, Conv2D, Dropout, MaxPooling2D, UpSampling2D, Reshape, Input
from tensorflow.keras.layers import LeakyReLU, BatchNormalization, LayerNormalization
from tensorflow.keras.models import Sequential
from tensorflow.keras.losses import mean_squared_error as mse
from tensorflow.keras.losses import BinaryCrossentropy
from tensorflow.keras.metrics import Mean
import datetime
import numpy as np
from tqdm import tqdm
import sys



def scaled_dot_product_attention(q, k, v):
  """Calculate the attention weights.
  q, k, v must have matching leading dimensions.
  k, v must have matching penultimate dimension, i.e.: seq_len_k = seq_len_v.
  The mask has different shapes depending on its type(padding or look ahead) 
  but it must be broadcastable for addition.
  
  Args:
    q: query shape == (..., seq_len_q, depth)
    k: key shape == (..., seq_len_k, depth)
    v: value shape == (..., seq_len_v, depth_v)
    mask: Float tensor with shape broadcastable 
          to (..., seq_len_q, seq_len_k). Defaults to None.
    
  Returns:
    output, attention_weights
  """

  matmul_qk = tf.matmul(q, k, transpose_b=True)  # (..., seq_len_q, seq_len_k)
  
  # scale matmul_qk
  dk = tf.cast(tf.shape(k)[-1], tf.float32)
  scaled_attention_logits = matmul_qk / tf.math.sqrt(dk)

  # add the mask to the scaled tensor.

  # softmax is normalized on the last axis (seq_len_k) so that the scores
  # add up to 1.
  attention_weights = tf.nn.softmax(scaled_attention_logits, axis=-1)  # (..., seq_len_q, seq_len_k)

  output = tf.matmul(attention_weights, v)  # (..., seq_len_q, depth_v)

  return output, attention_weights



class MultiHeadAttention(tf.keras.layers.Layer):
  def __init__(self, d_model, num_heads):
    super(MultiHeadAttention, self).__init__()
    self.num_heads = num_heads
    self.d_model = d_model
    
    assert d_model % self.num_heads == 0
    
    self.depth = d_model // self.num_heads
    
    self.wq = Dense(d_model)
    self.wk = Dense(d_model)
    self.wv = Dense(d_model)
    
    self.dense = Dense(d_model)
        
  def split_heads(self, x, batch_size):
    """Split the last dimension into (num_heads, depth).
    Transpose the result such that the shape is (batch_size, num_heads, seq_len, depth)
    """
    x = tf.reshape(x, (batch_size, -1, self.num_heads, self.depth))
    return tf.transpose(x, perm=[0, 2, 1, 3])
    
  def call(self, v, k, q):
    batch_size = tf.shape(q)[0]
    
    q = self.wq(q)  # (batch_size, seq_len, d_model)
    k = self.wk(k)  # (batch_size, seq_len, d_model)
    v = self.wv(v)  # (batch_size, seq_len, d_model)
    
    q = self.split_heads(q, batch_size)  # (batch_size, num_heads, seq_len_q, depth)
    k = self.split_heads(k, batch_size)  # (batch_size, num_heads, seq_len_k, depth)
    v = self.split_heads(v, batch_size)  # (batch_size, num_heads, seq_len_v, depth)
    
    # scaled_attention.shape == (batch_size, num_heads, seq_len_q, depth)
    # attention_weights.shape == (batch_size, num_heads, seq_len_q, seq_len_k)
    scaled_attention, attention_weights = scaled_dot_product_attention(q, k, v)
    scaled_attention = tf.transpose(scaled_attention, perm=[0, 2, 1, 3])  # (batch_size, seq_len_q, num_heads, depth)

    concat_attention = tf.reshape(scaled_attention, 
                                  (batch_size, -1, self.d_model))  # (batch_size, seq_len_q, d_model)

    output = self.dense(concat_attention)  # (batch_size, seq_len_q, d_model)
        
    return output, attention_weights


def point_wise_feed_forward_network(d_model, dff):
  return Sequential([
      Dense(dff, activation='relu'),  # (batch_size, seq_len, dff)
      Dense(d_model)  # (batch_size, seq_len, d_model)
  ])



class EncoderLayer(tf.keras.layers.Layer):
  def __init__(self, d_model, num_heads, dff, rate=0.1):
    super(EncoderLayer, self).__init__()

    self.mha = MultiHeadAttention(d_model, num_heads)
    self.ffn = point_wise_feed_forward_network(d_model, dff)

    self.layernorm1 = LayerNormalization(epsilon=1e-6)
    self.layernorm2 = LayerNormalization(epsilon=1e-6)
    
    self.dropout1 = Dropout(rate)
    self.dropout2 = Dropout(rate)
    
  def call(self, x, training):

    attn_output, _ = self.mha(x, x, x)  # (batch_size, input_seq_len, d_model)
    attn_output = self.dropout1(attn_output, training=training)
    out1 = self.layernorm1(x + attn_output)  # (batch_size, input_seq_len, d_model)
    
    ffn_output = self.ffn(out1)  # (batch_size, input_seq_len, d_model)
    ffn_output = self.dropout2(ffn_output, training=training)
    out2 = self.layernorm2(out1 + ffn_output)  # (batch_size, input_seq_len, d_model)
    
    return out2

class TransformerEncoder(tf.keras.layers.Layer):
  def __init__(self, num_layers, num_heads, d_model, d_input, dff, rate=0.1, is_pred=False, name='transformer_encoder'):
    super(TransformerEncoder, self).__init__()

    self.d_model = d_model
    self.num_layers = num_layers
    self.d_input = d_input
    self.is_pred = is_pred
    #self.embedding = tf.keras.layers.Embedding(input_vocab_size, d_model)
    #self.pos_encoding = positional_encoding(maximum_position_encoding, self.d_model)
    
    self.dense1 = Dense(d_model)
    self.dense2 = Dense(d_input)
    self.enc_layers = [EncoderLayer(self.d_model, num_heads, dff, rate) 
                       for _ in range(num_layers)]
  
    #self.dropout = tf.keras.layers.Dropout(rate)
        
  def call(self, x, training):

    # seq_len = tf.shape(x)[1]
    
    # adding embedding and position encoding.
    #x = self.embedding(x)  # (batch_size, input_seq_len, d_model)
    #x *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))
    #x += self.pos_encoding[:, :seq_len, :]

    #x = self.dropout(x, training=training)
    
    # convert d_input to d_model
    
    if self.is_pred == False:
        x = self.dense1(x) 
    
    for i in range(self.num_layers):
      x = self.enc_layers[i](x, training)
      if i == self.num_layers // 2:
          x_encoded = x
      
    # convert d_model to d_input
    x = self.dense2(x)
    
    return x, x_encoded  # (batch_size, input_seq_len, d_input), (batch_size, input_seq_len, d_model)

    
class Discriminator(tf.keras.Model):    
    def __init__(self, input_shape=(300,79,1), name='discriminator'):
        super(Discriminator, self).__init__()
        self.conv1 = Conv2D(4, (3,3), strides=(2, 1), padding='same', name='conv1')
        self.batch1 = BatchNormalization()
        self.act1 = LeakyReLU(alpha=0.1)
        self.conv2 = Conv2D(16, (3,3), strides=2, padding='same')
        self.batch2 = BatchNormalization()
        self.act2 = LeakyReLU(alpha=0.1)
        self.conv3 = Conv2D(64, (3,3), strides=2, padding='same')
        self.batch3 = BatchNormalization()
        self.act3 = LeakyReLU(alpha=0.1)
        self.conv4 = Conv2D(128, (3,3), strides=2, padding='same')
        self.batch4 = BatchNormalization()
        self.act4 = LeakyReLU(alpha=0.1)
        
        self.flat = Flatten()
        self.dense = Dense(1, activation='sigmoid')
        
    def call(self, pred_output):
        x = tf.expand_dims(pred_output, -1)
        x = self.conv1(x)
        x = self.batch1(x)
        x = self.act1(x)
        x = self.conv2(x)
        x = self.batch2(x)
        x = self.act2(x)
        x = self.conv3(x)
        x = self.batch3(x)
        x = self.act3(x)
        x = self.conv4(x)
        x = self.batch4(x)
        x = self.act4(x)
        x = self.flat(x)
        x = self.dense(x)
        return x
    
    
class cascaded_autoencoder(object):
    def __init__(self, len_time_window, batch_size=64):
        # sub-autoencoders
        self.NUM_LAYERS = 3
        self.NUM_HEADS = 4
        self.D_FF = 256
        self.LEN_TIME_WINDOW = len_time_window
        self.MODEL_DIR='./model/'
        self.LOG_DIR = './logs/'
        
        self.predictor = TransformerEncoder(self.NUM_LAYERS, self.NUM_HEADS, 64, 79, self.D_FF, is_pred=True, name='predictor')
        
        self.model = self._make_model() # model = s1, s2, s3, s4 + predictor
        self.discriminator = Discriminator(input_shape=(self.LEN_TIME_WINDOW,79,1), name='discriminator')

    def predict(self, input_X):       
        return self.model.predict(input_X)
    
    
    def _make_model(self):
        input_X = tf.keras.Input(shape=(self.LEN_TIME_WINDOW,79,))
        p_out = TransformerEncoder(num_layers=3, d_model=64, num_heads=4, d_input=79, dff=256, rate=0.1)(input_X)

        cascaded_autoencoder = tf.keras.Model(input_X, p_out, name='cascaded_autoencoder')
        return cascaded_autoencoder      
    
    def _calc_loss(self, target, predictions, fake=None): 
        
        loss_tr = tf.keras.losses.mean_squared_error(target, predictions)
        cross_entropy = BinaryCrossentropy(from_logits=False, label_smoothing=0.01)
        loss_anti_disc = cross_entropy(tf.zeros_like(fake), fake)
        
        loss_total = 10 * loss_tr + loss_anti_disc
        
        return loss_total, loss_tr, loss_anti_disc

    
    def  _calc_disc_loss(self, real, fake):
        cross_entropy = BinaryCrossentropy(from_logits=False, label_smoothing=0.01)
        real_loss = cross_entropy(tf.zeros_like(real), real) # real이 0인 loss
        fake_loss = cross_entropy(tf.ones_like(fake), fake) # fake가 1인 loss
        total_loss =  real_loss + fake_loss
        return total_loss        
        
        
    def _train_step(self, batch_X):
        with tf.GradientTape() as ae_tape, tf.GradientTape() as disc_tape:
            predictions = self.model(batch_X)
            target = batch_X
            output = predictions
            
            real_output = self.discriminator(target, training=True)
            fake_output = self.discriminator(output)
            
            total_loss, tr_loss, anti_disc_loss = self._calc_loss(target, predictions, fake_output)
            disc_loss = self._calc_disc_loss(real_output, fake_output)
            
        
        tf_gradients = ae_tape.gradient(total_loss, self.model.trainable_variables)
        disc_gradients = disc_tape.gradient(disc_loss, self.discriminator.trainable_variables)
        
        self.tf_optimizer.apply_gradients(zip(tf_gradients, self.model.trainable_variables))
        self.disc_optimizer.apply_gradients(zip(disc_gradients, self.discriminator.trainable_variables))
        
        train_losses = total_loss, disc_loss, tr_loss, anti_disc_loss
        return train_losses
        
    def train(self, train_data, n_epoch=10, ae_lr=1e-4, disc_lr=1e-3):
        TRAIN_LOG_DIR = self.LOG_DIR + 'train/' + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        train_summary_writer = tf.summary.create_file_writer(TRAIN_LOG_DIR)
        autoencoder_lr = ae_lr
        discriminator_lr = disc_lr
        
        import time
        self.tf_optimizer = tf.keras.optimizers.Adam(autoencoder_lr)
        self.disc_optimizer = tf.keras.optimizers.Adam(discriminator_lr)
        
        global_step = 0
        tf.print("Start Model Training at " + TRAIN_LOG_DIR[13:], output_stream=sys.stdout) # Print "Start Time"
        start_time = time.time()
        for epoch in range(n_epoch):            
            loss_total = Mean()
            loss_tr = Mean()
            loss_disc = Mean()
            loss_anti_disc = Mean()
            
            for step, batch in enumerate(train_data):
                batch_x = tf.cast(batch, tf.float32)
                # batch_y = tf.cast(batch[1], tf.float32)
                
                # train one onestep
                train_losses = self._train_step(batch_x)
                total_loss, disc_loss, tr_loss, anti_disc_loss = train_losses
                loss_total(total_loss)
                loss_tr(tr_loss)
                loss_disc(disc_loss)
                loss_anti_disc(anti_disc_loss)
                
                # Log every 1000 steps
                if global_step % 1000 == 0:       
                    template = 'Epoch {}, Step {}, collapse {} \n Total loss: {} \n '
                    template += 'TR Loss {} \n Anti_disc_loss {} \n DISC loss: {} \n '
                    
                    tf.print(template.format(epoch+1, step,
                                          round(time.time()-start_time, 3),
                                          loss_total.result(),
                                          loss_tr.result(),
                                          loss_anti_disc.result(),
                                          loss_disc.result(),
                                          output_stream=sys.stdout))
        
                    with train_summary_writer.as_default():
                        tf.summary.scalar('loss_Total', loss_total.result(), step=global_step)
                        tf.summary.scalar('loss_TR', loss_tr.result(), step=global_step)
                        tf.summary.scalar('loss_Anti_DISC', loss_anti_disc.result(), step=global_step)
                        tf.summary.scalar('loss_DISC', loss_disc.result(), step=global_step)
                    # clear loss state
                    loss_total.reset_states()
                    loss_tr.reset_states()
                    loss_anti_disc.reset_states()
                    loss_disc.reset_states()
                    
                    # end of if global_step % 1000 == 0:
                global_step += 1
            # end of for step, batch in enumerate(train_data):
        # end of for epoch in range(n_epoch):    
        print("Training is done. Elapsed time : {}".format(round(time.time()-start_time, 3)))
    
    def save(self):
        self.model.save_weights(self.MODEL_DIR + 'cas_ae/' + 'model.ckpt')
        self.discriminator.save_weights(self.MODEL_DIR + 'discriminator/' + 'model.ckpt')
        
    def restore(self):
        self.model.load_weights(self.MODEL_DIR + 'cas_ae/' + 'model.ckpt')
        self.discriminator.load_weights(self.MODEL_DIR + 'discriminator/' + 'model.ckpt')
                        
    def evaluate(self, test_data, test_label):
        y_pred_disc = np.empty(0)
        y_pred_tr1 = np.empty(0)
        y_true = np.empty(0)
        tqdm.write('Generating predictions...')
        for batch_x, batch_label in tqdm(zip(test_data, test_label)):
            batch_x = tf.cast(batch_x, tf.float32)
            # get prediction of discriminator
            disc_prediction = self.discriminator(batch_x, training=False)
            y_pred_disc = np.append(y_pred_disc, disc_prediction.numpy())
            
            # get prediction of autoencoder 
            ae_prediction = self.model(batch_x) # [batch, win_len, cols]
            err = np.square(batch_x.numpy() - ae_prediction.numpy())
            err = err.reshape([err.shape[0], -1]).mean(1)
            y_pred_tr1 = np.append(y_pred_tr1, err)
            
            # get labels
            batch_label = tf.reduce_max(batch_label, axis=1) # if one or more label in time_window is 1, label is 1
            if len(y_true) == 0:
                y_true = batch_label.numpy()
            else:
                y_true = np.vstack((y_true, batch_label.numpy()))
        
        print(y_pred_disc)
        return (y_pred_disc, y_pred_tr1, y_true)