

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
from transformer_classes import Transformer

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

def create_masks(inp, tar):
    # Encoder padding mask
    enc_padding_mask = create_padding_mask(inp)

    # Used in the 2nd attention block in the decoder.
    # This padding mask is used to mask the encoder outputs.
    dec_padding_mask = create_padding_mask(inp)

    # Used in the 1st attention block in the decoder.
    # It is used to pad and mask future tokens in the input received by
    # the decoder.
    look_ahead_mask = create_look_ahead_mask(tf.shape(tar)[1])
    dec_target_padding_mask = create_padding_mask(tar)
    combined_mask = tf.maximum(dec_target_padding_mask, look_ahead_mask)

    return enc_padding_mask, combined_mask, dec_padding_mask

class CustomSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, d_model, warmup_steps=4000):
        super(CustomSchedule, self).__init__()

        self.d_model = d_model
        self.d_model = tf.cast(self.d_model, tf.float32)

        self.warmup_steps = warmup_steps

    def __call__(self, step):
        arg1 = tf.math.rsqrt(step)
        arg2 = step * (self.warmup_steps ** -1.5)

        return tf.math.rsqrt(self.d_model) * tf.math.minimum(arg1, arg2)

def loss_function(real, pred):

    loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True, reduction='none')

    mask = tf.math.logical_not(tf.math.equal(real, 0)) #Where there ar nonzero
    loss_ = loss_object(real, pred)

    mask = tf.cast(mask, dtype=loss_.dtype)
    loss_ *= mask

    return tf.reduce_sum(loss_)/tf.reduce_sum(mask)


def accuracy_function(real, pred):
    real = tf.cast(real, dtype=tf.int64)
    pred = tf.cast(pred, dtype=tf.int64)
    accuracies = tf.equal(real, pred)

    mask = tf.math.logical_not(tf.math.equal(real, 0))
    accuracies = tf.math.logical_and(mask, accuracies)
    accuracies = tf.cast(accuracies, dtype=tf.float32)
    mask = tf.cast(mask, dtype=tf.float32)
    return tf.reduce_sum(accuracies)/tf.reduce_sum(mask) #Normalize loss with the number of items in mask (nonzero)

def evaluate(sequence, transformer, max_length=202):
    '''Evaluate the translations
    '''

    encoder_input = tf.convert_to_tensor([sequence])

    #Decoder output start and end tokens
    start, end = 7, 8
    output = tf.convert_to_tensor([start],dtype=tf.int64) #The decoder start token is omitted here
    output = tf.expand_dims(output, 0)

    for i in range(max_length):
        enc_padding_mask, combined_mask, dec_padding_mask = create_masks(
            encoder_input, output)

        # predictions.shape == (batch_size, seq_len, vocab_size)
        predictions, attention_weights = transformer(encoder_input,
                                                     output,
                                                     False,
                                                     enc_padding_mask,
                                                     combined_mask,
                                                     dec_padding_mask)

        # select the last word from the seq_len dimension
        predictions = predictions[:, -1:, :]  # (batch_size, 1, vocab_size)

        predicted_id = tf.argmax(predictions, axis=-1)

        # concatentate the predicted_id to the output which is given to the decoder
        # as its input.
        output = tf.concat([output, predicted_id], axis=-1)

        # return the result if the predicted_id is equal to the end token
        if predicted_id == end:
            break

    output = output.numpy()[0,:]

    return output, attention_weights

def create_and_train_model(EPOCHS, batch_size, maxlen, input_vocab_size, target_vocab_size, d_model,num_heads, dff,num_layers, x_train, x_valid):
    '''Create the transformer model
    '''

    dropout_rate=0.5
    #Transformer
    transformer = Transformer(
    num_layers=num_layers,
    d_model=d_model,
    num_heads=num_heads,
    dff=dff,
    input_vocab_size=input_vocab_size,
    target_vocab_size=target_vocab_size,
    pe_input=maxlen+2,
    pe_target=maxlen+2,
    rate=dropout_rate)


    learning_rate = CustomSchedule(d_model)

    optimizer = tf.keras.optimizers.Adam(learning_rate, beta_1=0.9, beta_2=0.98,
                                         epsilon=1e-9)


    train_loss = tf.keras.metrics.Mean(name='train_loss')
    train_accuracy = tf.keras.metrics.Mean(name='train_accuracy')
    valid_loss = tf.keras.metrics.Mean(name='valid_loss')
    valid_accuracy = tf.keras.metrics.Mean(name='valid_accuracy')
    # The @tf.function trace-compiles train_step into a TF graph for faster
    # execution. The function specializes to the precise shape of the argument
    # tensors. To avoid re-tracing due to the variable sequence lengths or variable
    # batch sizes (the last batch is smaller), use input_signature to specify
    # more generic shapes.

    train_step_signature = [
        tf.TensorSpec(shape=(None, None), dtype=tf.int64),
        tf.TensorSpec(shape=(None, None), dtype=tf.int64),
    ]


    @tf.function(input_signature=train_step_signature)
    def train_step(inp, tar):
        tar_inp = tar[:, :-1] #The first char is the start token - the last the end token
        tar_real = tar[:, 1:]

        enc_padding_mask, combined_mask, dec_padding_mask = create_masks(inp, tar_inp)

        with tf.GradientTape() as tape:
            predictions, _ = transformer(inp, tar_inp,
                                         True,
                                         enc_padding_mask,
                                         combined_mask,
                                         dec_padding_mask)


            #Here a decoder is added to get the accuracy of the allowed transitions - this should really be linked to the loss as well
            argmax_predictions = tf.argmax(predictions,axis=2)
            #Get pred start - the type
            #1=no targeting peptide/Inside cell, 2=sp: signal peptide, 3=mt:mitochondrial transit peptide,
            #4=ch:chloroplast transit peptide, 5=th:thylakoidal lumen composite transit peptide
            #6=Outside of cell - only valid for SPs - not for the peptides going into mt or ch/th
            t1 = argmax_predictions[:,1]
            #The ones that start with 1 should continue with 1 up to the mask
            #The others should have the same char as in t1 up to the point of the first 1 or 6
            #Here all predicted annotations are set to the first predicted character up to the point
            #where they differ
            for i in range(batch_size):
                mi = tf.keras.backend.min(tf.where(argmax_predictions[i,1:]!=t1[i])) #Get mismatch index
                indices = tf.range(1,mi,1) #(start, limit, delta)
                inserts = tf.repeat(t1[i],mi-1)
                tf.tensor_scatter_nd_update(argmax_predictions[i],indices, inserts)

            loss = loss_function(tar_real, predictions)

            gradients = tape.gradient(loss, transformer.trainable_variables)
            optimizer.apply_gradients(zip(gradients, transformer.trainable_variables))

            train_loss(loss)
            train_accuracy(accuracy_function(tar_real, argmax_predictions))

    @tf.function()
    def valid_step(inp, tar):
        tar_inp = tar[:, :-1] #The first char is the start token - the last the end token
        tar_real = tar[:, 1:]
        enc_padding_mask, combined_mask, dec_padding_mask = create_masks(inp, tar_inp)
        predictions, _ = transformer(inp, tar_inp, True, enc_padding_mask, combined_mask, dec_padding_mask)
        loss = loss_function(tar_real, predictions)

        argmax_predictions = tf.argmax(predictions,axis=2)
        #Get pred start - the type
        #1=no targeting peptide/Inside cell, 2=sp: signal peptide, 3=mt:mitochondrial transit peptide,
        #4=ch:chloroplast transit peptide, 5=th:thylakoidal lumen composite transit peptide
        #6=Outside of cell - only valid for SPs - not for the peptides going into mt or ch/th
        t1 = argmax_predictions[:,1]
        #The ones that start with 1 should continue with 1 up to the mask
        #The others should have the same char as in t1 up to the point of the first 1 or 6
        #Here all predicted annotations are set to the first predicted character up to the point
        #where they differ
        for i in range(len(x_valid[0])):
            mi = tf.keras.backend.min(tf.where(argmax_predictions[i,1:]!=t1[i])) #Get mismatch index
            indices = tf.range(1,mi,1) #(start, limit, delta)
            inserts = tf.repeat(t1[i],mi-1)
            tf.tensor_scatter_nd_update(argmax_predictions[i],indices, inserts)

        valid_loss(loss)
        valid_accuracy(accuracy_function(tar_real, argmax_predictions))


    #Number of steps per epoch
    steps_per_epoch = int(len(x_train[0])/batch_size-1)
    train_inds = np.arange(len(x_train[0])) #Indices

    for epoch in range(EPOCHS):
        train_loss.reset_states()
        train_accuracy.reset_states()
        valid_loss.reset_states()
        valid_accuracy.reset_states()

        #Shuffle inds
        np.random.shuffle(train_inds)

        # inp -> portuguese, tar -> english
        for batch in range(steps_per_epoch):
            batch_inds = train_inds[batch:batch+batch_size]
            inp, orgs, tar = x_train[0][batch_inds], x_train[1][batch_inds], x_train[2][batch_inds]
            train_step(inp, tar)

        print(f'Epoch {epoch + 1} Loss {train_loss.result():.4f} Accuracy {train_accuracy.result():.4f}')

        #Evaluate the valid set
        #valid_step(x_valid[0], x_valid[2])
        #print(f'Epoch {epoch + 1} Valid Loss {valid_loss.result():.4f}  Valid Accuracy {valid_accuracy.result():.4f}')


    #Evaluate the model on the valid set
    output, attention_weights = evaluate(x_valid[0][50], transformer, 202)
    pdb.set_trace()
