#! /usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import sys
sys.path.insert(0, "../")
import numpy as np
import pandas as pd
import time
from collections import Counter
#Preprocessing and evaluation
from process_data import parse_and_format
#Keras
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from categorical_focal_loss import SparseCategoricalFocalLoss
from tensorflow.keras.callbacks import ModelCheckpoint
from attention_class import MultiHeadSelfAttention

from tensorflow.keras.models import model_from_json
import glob


import pdb

#Arguments for argparse module:
parser = argparse.ArgumentParser(description = '''Obtain attention activations.''')
parser.add_argument('--checkpointdir', nargs=1, type= str, default=sys.stdin, help = '''path checkpoints with .h5 files containing weights for net.''')
parser.add_argument('--datadir', nargs=1, type= str, default=sys.stdin, help = 'Path to data directory.')
parser.add_argument('--test_partition', nargs=1, type= int, default=sys.stdin, help = 'Which CV fold to test/bench on.')
parser.add_argument('--valid_partition', nargs=1, type= int, default=sys.stdin, help = 'Which CV fold to get the valid model from.')
parser.add_argument('--variable_params', nargs=1, type= str, default=sys.stdin, help = 'Path to csv with variable params.')
parser.add_argument('--param_combo', nargs=1, type= int, default=sys.stdin, help = 'Parameter combo.')
parser.add_argument('--outdir', nargs=1, type= str, default=sys.stdin, help = 'Output directory.')


#FUNCTIONS
class TokenAndPositionEmbedding(layers.Layer):
    def __init__(self, maxlen, vocab_size, embed_dim):
        super(TokenAndPositionEmbedding, self).__init__()
        self.token_emb = layers.Embedding(input_dim=vocab_size, output_dim=embed_dim)
        self.pos_emb = layers.Embedding(input_dim=maxlen, output_dim=embed_dim)

    def call(self, x):
        maxlen = tf.shape(x)[-1]
        positions = tf.range(start=0, limit=maxlen, delta=1)
        positions = self.pos_emb(positions)
        x = self.token_emb(x)
        return x + positions

class EncoderBlock(layers.Layer):
    def __init__(self, embed_dim, num_heads, ff_dim, rate=0.1):
        super(EncoderBlock, self).__init__()
        self.att = MultiHeadSelfAttention(embed_dim,num_heads)
        self.ffn = keras.Sequential(
            [layers.Dense(ff_dim, activation="relu"), layers.Dense(embed_dim),]
        )
        self.layernorm1 = layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = layers.Dropout(rate)
        self.dropout2 = layers.Dropout(rate)

    def call(self, in_q,in_k,in_v, training): #Inputs is a list with [q,k,v]
        attn_output,attn_weights = self.att(in_q,in_k,in_v) #The weights are needed for downstream analysis
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(in_q + attn_output)
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        return self.layernorm2(out1 + ffn_output), attn_weights

class DecoderBlock(layers.Layer):
    def __init__(self, embed_dim, num_heads, ff_dim, rate=0.1):
        super(DecoderBlock, self).__init__()
        self.att = MultiHeadSelfAttention(embed_dim,num_heads)
        self.ffn = keras.Sequential(
            [layers.Dense(ff_dim, activation="relu"), layers.Dense(embed_dim),]
        )
        self.layernorm1 = layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = layers.Dropout(rate)
        self.dropout2 = layers.Dropout(rate)

    def call(self, in_q,in_k,in_v, training): #Inputs is a list with [q,k,v]
        #Self-attention
        attn_output1,attn_weights1 = self.att(in_q,in_q,in_q) #The weights are needed for downstream analysis
        attn_output1 = self.dropout1(attn_output1, training=training)
        out1 = self.layernorm1(in_v + attn_output1)
        #Encoder-decoder attention
        attn_output2,attn_weights2 = self.att(out1,in_k,in_v) #The weights are needed for downstream analysis
        attn_output2 = self.dropout1(attn_output2, training=training)
        out2 = self.layernorm1(attn_output2 + attn_output1)
        ffn_output = self.ffn(out2)
        ffn_output = self.dropout2(ffn_output, training=training)
        return self.layernorm2(out2 + ffn_output), attn_weights2

def create_model(maxlen, vocab_size, embed_dim,num_heads, ff_dim,num_layers,num_iterations):
    '''Create the transformer model
    '''

    seq_input = layers.Input(shape=(maxlen,)) #Input aa sequences
    seq_target = layers.Input(shape=(maxlen,)) #Targets - annotations
    kingdom_input = layers.Input(shape=(maxlen,4)) #4 kingdoms, Archaea, Eukarya, Gram +, Gram -

    ##Embeddings
    embedding_layer1 = TokenAndPositionEmbedding(maxlen, vocab_size, embed_dim)
    embedding_layer2 = TokenAndPositionEmbedding(maxlen, 6, embed_dim+4) #Need to add 4 so that x1 and x2 match
    x1 = embedding_layer1(seq_input)
    #Add kingdom input
    x1 = layers.Concatenate()([x1,kingdom_input])
    x2 = embedding_layer2(seq_target)

    #Define the transformer
    encoder = EncoderBlock(embed_dim+4, num_heads, ff_dim)
    decoder = DecoderBlock(embed_dim+4, num_heads, ff_dim)
    #Iterate
    for i in range(num_iterations):
        #Encode
        for j in range(num_layers):
            x1, enc_attn_weights = encoder(x1,x1,x1) #q,k,v
        #Decoder
        for k in range(num_layers):
            x2, enc_dec_attn_weights = decoder(x2,x1,x1) #q,k,v - the k and v from the encoder goes into he decoder

        x2 = layers.Dense(6, activation="softmax")(x2) #Annotate
        x_rs = layers.Reshape((maxlen,6))(x2)
        x2 = tf.math.argmax(x_rs,axis=-1) #Needed for iterative training
        x2 = embedding_layer2(x2)

    x2, enc_dec_attn_weights = decoder(x2,x1,x1) #q,k,v - the k and v from the encoder goes into he decoder
    preds = layers.Dense(6, activation="softmax")(x2) #Annotate
    #preds = layers.Reshape((maxlen,6),name='annotation')(x2)
    #pred_type = layers.Dense(4, activation="softmax",name='type')(x) #Type of protein
    #pred_cs = layers.Dense(1, activation="elu", name='pred_cs')(x)


    model = keras.Model(inputs=[seq_input,seq_target,kingdom_input], outputs=preds)

    return model


def load_model(variable_params, param_combo, weights):

    #Params
    net_params = variable_params.loc[param_combo-1]
    #Fixed params
    vocab_size = 21  # Only consider the top 20k words
    maxlen = 70  # Only consider the first 70 amino acids
    #Variable params
    embed_dim = int(net_params['embed_dim']) #32  # Embedding size for each token
    num_heads = int(net_params['num_heads']) #1  # Number of attention heads
    ff_dim = int(net_params['ff_dim']) #32  # Hidden layer size in feed forward network inside transformer
    num_layers = int(net_params['num_layers']) #1  # Number of attention heads
    batch_size = int(net_params['batch_size']) #32
    num_iterations = int(net_params['num_iterations'])
    #Create model
    model = create_model(maxlen, vocab_size, embed_dim,num_heads, ff_dim,num_layers,num_iterations)
    model.load_weights(weights)
    print(model.summary())

    return model


def get_data(datadir, test_partition, maxlen):
    '''Get the test data
    '''

    #Get the bench data
    bench_meta = pd.read_csv(datadir+'bench_meta.csv')
    bench_seqs = np.load(datadir+'bench_seqs.npy',allow_pickle=True)
    bench_annotations = np.load(datadir+'bench_annotations.npy',allow_pickle=True)


    bench_kingdoms = bench_meta.Kingdom.values
    bench_meta['Type'] = bench_meta['Type'].replace({'NO_SP':0,'SP':1,'TAT':2,'LIPO':3})
    bench_types = bench_meta.Type.values
    #Onehot conversion
    bench_kingdoms = np.eye(4)[bench_kingdoms]

    #Get data
    #Run through all by taking as input
    bench_i = bench_meta[bench_meta.Partition==test_partition].index

    #bench
    x_bench_seqs = bench_seqs[bench_i]
    x_bench_kingdoms = bench_kingdoms[bench_i]
    x_bench_kingdoms = np.repeat(np.expand_dims(x_bench_kingdoms,axis=1),70,axis=1)
    #Random annotations
    x_bench_target_inp =  np.random.randint(6,size=(len(bench_i),maxlen))
    x_bench = [x_bench_seqs,x_bench_target_inp,x_bench_kingdoms]
    y_bench = [bench_annotations[bench_i],bench_types[bench_i]]

    return x_bench, y_bench


def get_attention(model,data):
    '''Obtain the output of the attention layers
    '''
    # with a Sequential model
    #Enc self-attention
    get_enc_layer_output = keras.backend.function([model.layers[0].input,model.layers[2].input],[model.layers[4].output])
    enc_attention = get_enc_layer_output([data[0],data[2]])[0][0][1]

    #Enc-dec attention
    get_dec_layer_output = keras.backend.function([model.layers[0].input,model.layers[5].input,model.layers[2].input],[model.layers[7].output])
    enc_dec_attention = get_dec_layer_output(data)[0][0][1]
    
    return enc_attention, enc_dec_attention

######################MAIN######################
args = parser.parse_args()
checkpointdir=args.checkpointdir[0]
datadir = args.datadir[0]
test_partition = args.test_partition[0]
valid_partition = args.valid_partition[0]
variable_params=pd.read_csv(args.variable_params[0])
param_combo=args.param_combo[0]
outdir = args.outdir[0]

#Load and run model

#Get data
x_bench, y_bench = get_data(datadir, test_partition,70)
#Get activations - printed to stderr in multi_head_attention function
#weights
weights=glob.glob(checkpointdir+'TP'+str(test_partition)+'/vp'+str(valid_partition)+'/*.hdf5')

#model
model = load_model(variable_params, param_combo, weights[0])
#Get attention
enc_attention, enc_dec_attention = get_attention(model,x_bench)
#Save
np.save(outdir+'enc_attention_'+str(test_partition)+'_'+str(valid_partition)+'.npy',enc_attention)
np.save(outdir+'enc_dec_attention_'+str(test_partition)+'_'+str(valid_partition)+'.npy',enc_dec_attention)
#Save the true annotations and types
np.save(outdir+'annotations_'+str(test_partition)+'.npy',y_bench[0])
np.save(outdir+'types_'+str(test_partition)+'.npy',y_bench[1])
#Save the sequences
np.save(outdir+'seqs_'+str(test_partition)+'.npy',x_bench[0])
#Save the kingdoms
np.save(outdir+'kingdoms_'+str(test_partition)+'.npy',x_bench[2])

#Save pred annotations
pred_annotations = model.predict(x_bench)
np.save(outdir+'pred_annotations_'+str(test_partition)+'_'+str(valid_partition)+'.npy',pred_annotations)
