

import argparse
import sys
import os
import numpy as np
import pandas as pd
import time
from collections import Counter

#Keras
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

#Custom
from categorical_focal_loss import SparseCategoricalFocalLoss

import pdb

#set_session(sess)  # set this TensorFlow session as the default session for Keras

#####FUNCTIONS and CLASSES#####
def create_padding_mask(seq):
    '''Mask all the pad tokens in the batch of sequence.
    It ensures that the model does not treat padding as the input.
    The mask indicates where pad value 0 is present: it outputs a 1 at those locations, and a 0 otherwise.
    '''
    seq = tf.cast(tf.math.equal(seq, 0), tf.float32) #Returns the positions equal to 0 = padded

    # add extra dimensions to add the padding
    # to the attention logits.
    return seq[:, tf.newaxis, tf.newaxis, :]  # (batch_size, 1, 1, seq_len)

def create_look_ahead_mask(size):
    '''The look-ahead mask is used to mask the future tokens in a sequence.
    In other words, the mask indicates which entries should not be used.
    This means that to predict the third word, only the first and second word will be used.
    Similarly to predict the fourth word, only the first, second and the third word will be used and so on.
    Here - the previous predictions will be used and not the true "translations"
    This is due to the lack of classes and thereby ease for the model to "cheat"
    if it knows only one of the true TP characters
    '''
    mask = 1 - tf.linalg.band_part(tf.ones((size, size)), -1, 0)
    return mask  # (seq_len, seq_len)

def create_model(maxlen, vocab_size, embed_dim,num_heads, ff_dim,num_layers, find_lr):
    '''Create the transformer model
    '''

    seq_input = layers.Input(shape=(maxlen,)) #Input aa sequences
    seq_target = layers.Input(shape=(maxlen,)) #Targets - annotations
    org_input = layers.Input(shape=(maxlen,2)) #2 Organisms plant/not

    ##Embeddings
    embedding_layer1 = TokenAndPositionEmbedding(maxlen, vocab_size, embed_dim)
    embedding_layer2 = TokenAndPositionEmbedding(maxlen, 1, embed_dim+2) #5 annotation classes. Need to add 2 so that x1 and x2 match - the organsims
    x1 = embedding_layer1(seq_input)
    #Add kingdom input
    x1 = layers.Concatenate()([x1,org_input])
    x2 = embedding_layer2(seq_target)

    #Define the transformer
    encoder = EncoderBlock(embed_dim+2, num_heads, ff_dim)
    decoder = DecoderBlock(embed_dim+2, num_heads, ff_dim)

    #Encode
    for j in range(num_layers):
        x1, enc_attn_weights = encoder(x1,x1,x1) #q,k,v
    #Decoder
    for k in range(num_layers):
        x2, enc_dec_attn_weights = decoder(x2,x1,x1) #q,k,v - the k and v from the encoder goes into he decoder

    x2 = tf.keras.layers.GlobalMaxPooling1D( data_format='channels_first')(x2)
    pred_CS = layers.Dense(maxlen, activation="softmax", name='CS')(x2) # #CS
    pred_type = layers.Dense(5, activation="softmax", name='Type')(x2) #Type


    model = keras.Model(inputs=[seq_input,seq_target,org_input], outputs=[pred_CS,pred_type])

    #learning_rate
    initial_learning_rate = 0.001 #From lr finder
    if find_lr ==False:
        lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate,
        decay_steps=10000,
        decay_rate=0.96,
        staircase=True)
    else:
        lr_schedule = initial_learning_rate

    #Optimizer
    opt = tf.keras.optimizers.Adam(learning_rate = lr_schedule,amsgrad=True)

    #Compile
    #The CS loss should probably be scaled - much harder to learn: x4 in paper more or less
    model.compile(optimizer = opt, loss= [SparseCategoricalFocalLoss(gamma=3),SparseCategoricalFocalLoss(gamma=3)],
                                metrics=["accuracy"])

    return model
