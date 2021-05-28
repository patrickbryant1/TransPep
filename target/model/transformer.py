

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
from categorical_focal_loss import SparseCategoricalFocalLoss
from tensorflow.keras.callbacks import ModelCheckpoint


#visualization
from tensorflow.keras.callbacks import TensorBoard

#Custom
from process_data import parse_and_format
from attention_class import MultiHeadSelfAttention
#from lr_finder import LRFinder


import pdb
#Arguments for argparse module:
parser = argparse.ArgumentParser(description = '''A Transformer Neural Network for sorting peptides.''')

parser.add_argument('--datadir', nargs=1, type= str, default=sys.stdin, help = 'Path to data directory.')
parser.add_argument('--variable_params', nargs=1, type= str, default=sys.stdin, help = 'Path to csv with variable params.')
parser.add_argument('--param_combo', nargs=1, type= int, default=sys.stdin, help = 'Parameter combo.')
parser.add_argument('--checkpointdir', nargs=1, type= str, default=sys.stdin, help = 'Path to checpoint directory. Include /in end')
parser.add_argument('--save_model', nargs=1, type= int, default=sys.stdin, help = 'If to save model or not: 1= True, 0 = False')
parser.add_argument('--checkpoint', nargs=1, type= int, default=sys.stdin, help = 'If to checkpoint or not: 1= True, 0 = False')
parser.add_argument('--num_epochs', nargs=1, type= int, default=sys.stdin, help = 'Num epochs (int)')
parser.add_argument('--outdir', nargs=1, type= str, default=sys.stdin, help = 'Path to output directory. Include /in end')

#from tensorflow.keras.backend import set_session
#config = tf.ConfigProto()
#config.gpu_options.allow_growth = True  # dynamically grow the memory used on the GPU
#config.log_device_placement = True  # to log device placement (on which device the operation ran)
                                    # (nothing gets printed in Jupyter, only if you run it standalone)
#sess = tf.Session(config=config)
#set_session(sess)  # set this TensorFlow session as the default session for Keras

#####FUNCTIONS and CLASSES#####
#####FUNCTIONS and CLASSES#####
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
        out1 = self.layernorm1(in_q + attn_output1)
        #Encoder-decoder attention
        attn_output2,attn_weights2 = self.att(out1,in_k,in_v) #The weights are needed for downstream analysis
        attn_output2 = self.dropout1(attn_output2, training=training)
        out2 = self.layernorm1(attn_output2 + out1)
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
        x_rs = layers.Reshape((maxlen,7))(x2)
        x2 = tf.math.argmax(x_rs,axis=-1) #Needed for iterative training
        x2 = embedding_layer2(x2)

    x2, enc_dec_attn_weights = decoder(x2,x1,x1) #q,k,v - the k and v from the encoder goes into he decoder
    preds = layers.Dense(6, activation="softmax")(x2) #Annotate
    #preds = layers.Reshape((maxlen,6),name='annotation')(x2)
    #pred_type = layers.Dense(4, activation="softmax",name='type')(x) #Type of protein
    #pred_cs = layers.Dense(1, activation="elu", name='pred_cs')(x)


    model = keras.Model(inputs=[seq_input,seq_target,kingdom_input], outputs=preds)
    #Optimizer
    initial_learning_rate = 0.001
    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate,
    decay_steps=10000,
    decay_rate=0.96,
    staircase=True)

    opt = tf.keras.optimizers.Adam(learning_rate = lr_schedule,amsgrad=True)

    #Compile
    model.compile(optimizer = opt, loss= SparseCategoricalFocalLoss(gamma=2), metrics=["accuracy"])

    return model

######################MAIN######################
args = parser.parse_args()
datadir = args.datadir[0]
#Get parameters
variable_params=pd.read_csv(args.variable_params[0])
param_combo = args.param_combo[0]
checkpointdir = args.checkpointdir[0]
save_model = bool(args.save_model[0])
checkpoint = bool(args.checkpoint[0])
num_epochs = args.num_epochs[0]
outdir = args.outdir[0]

#Params
net_params = variable_params.loc[param_combo-1]
test_partition = int(net_params['test_partition'])
#Fixed params
vocab_size = 21  #Amino acids and unknown (X)
maxlen = 200  # Only consider the first 70 amino acids


#Get data
try:
    meta = pd.read_csv(datadir+'meta.csv')
    annotations = np.load(datadir+'annotations.npy', allow_pickle=True)
    sequences = np.load(datadir+'sequences.npy', allow_pickle=True)

except:
    data = np.load(datadir+'targetp_data.npz') #'x', 'y_cs', 'y_type', 'len_seq', 'org', 'fold', 'ids'
    meta,  sequences, annotations = parse_and_format(datadir+'targetp.fasta',data)
    #Save
    meta.to_csv(datadir+'meta.csv',index=False)
    np.save(datadir+'annotations.npy',annotations)
    np.save(datadir+'sequences.npy',sequences)



#Get data
#Run through all by taking as input
# Nested cross-validation loop with 5 folds from https://github.com/JJAlmagro/TargetP-2.0/blob/master/train.py
test_i = np.where(meta.Fold ==test_partition)[0]
train_losses = []
valid_losses = []
inner_partitions_interval = np.setdiff1d(np.arange(5),test_partition)
for valid_partition in np.setdiff1d(np.arange(5),test_partition):
    print('Validation partition',valid_partition)
    valid_i = np.where(meta.Fold==valid_partition)[0]
    train_i = np.setdiff1d(np.arange(len(meta)),np.concatenate([test_i,valid_i]))

    #Training data
    x_train_seqs = sequences[train_i]
    x_train_orgs = np.repeat(np.expand_dims(meta.Org[train_i],axis=1),maxlen,axis=1)
    #Random annotations are added as input
    x_train_target_inp = np.random.randint(5,size=(len(train_i),maxlen))
    x_train = [x_train_seqs,x_train_target_inp,x_train_orgs] #inp seq, target annoation, organism
    y_train = annotations[train_i] #,train_types[train_i]]

    #Validation data
    x_valid_seqs = sequences[valid_i]
    x_valid_orgs = np.repeat(np.expand_dims(meta.Org[valid_i],axis=1),maxlen,axis=1)
    #Random annotations are added as input
    x_valid_target_inp = np.random.randint(5,size=(len(valid_i),maxlen))
    x_valid = [x_valid_seqs,x_valid_target_inp,x_valid_orgs] #inp seq, target annoation, organism
    y_valid = annotations[valid_i] #,train_types[train_i]]

    pdb.set_trace()
