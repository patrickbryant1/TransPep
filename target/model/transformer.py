

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
from lr_finder import LRFinder
#from lr_finder import LRFinder


import pdb
#Arguments for argparse module:
parser = argparse.ArgumentParser(description = '''A Transformer Neural Network for sorting peptides.''')

parser.add_argument('--datadir', nargs=1, type= str, default=sys.stdin, help = 'Path to data directory.')
parser.add_argument('--variable_params', nargs=1, type= str, default=sys.stdin, help = 'Path to csv with variable params.')
parser.add_argument('--param_combo', nargs=1, type= int, default=sys.stdin, help = 'Parameter combo.')
parser.add_argument('--checkpointdir', nargs=1, type= str, default=sys.stdin, help = 'Path to checpoint directory. Include /in end')
parser.add_argument('--checkpoint', nargs=1, type= int, default=sys.stdin, help = 'If to checkpoint or not: 1= True, 0 = False')
parser.add_argument('--num_epochs', nargs=1, type= int, default=sys.stdin, help = 'Num epochs (int)')
parser.add_argument('--find_lr', nargs=1, type= int, default=sys.stdin, help = 'Find lr (1) or not (0)')
parser.add_argument('--outdir', nargs=1, type= str, default=sys.stdin, help = 'Path to output directory. Include /in end')

#from tensorflow.keras.backend import set_session
#config = tf.ConfigProto()
#config.gpu_options.allow_growth = True  # dynamically grow the memory used on the GPU
#config.log_device_placement = True  # to log device placement (on which device the operation ran)
                                    # (nothing gets printed in Jupyter, only if you run it standalone)
#sess = tf.Session(config=config)
#set_session(sess)  # set this TensorFlow session as the default session for Keras

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
    initial_learning_rate = 0.001
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
    model.compile(optimizer = opt, loss= [SparseCategoricalFocalLoss(gamma=2),SparseCategoricalFocalLoss(gamma=2)], metrics=["accuracy"])

    return model

######################MAIN######################
args = parser.parse_args()
datadir = args.datadir[0]
#Get parameters
variable_params=pd.read_csv(args.variable_params[0])
param_combo = args.param_combo[0]
checkpointdir = args.checkpointdir[0]
checkpoint = bool(args.checkpoint[0])
num_epochs = args.num_epochs[0]
find_lr = bool(args.find_lr[0])
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
    CS = meta.CS.values
    Types = meta.Type.values
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
    x_train_orgs = np.repeat(np.expand_dims(np.eye(2)[meta.Org[train_i]],axis=1),maxlen,axis=1)
    #Random annotations are added as input
    x_train_target_inp = np.random.randint(1,size=(len(train_i),maxlen))
    x_train = [x_train_seqs,x_train_target_inp,x_train_orgs] #inp seq, target annoation, organism
    y_train = [CS[train_i],Types[train_i]]

    #Validation data
    x_valid_seqs = sequences[valid_i]
    x_valid_orgs = np.repeat(np.expand_dims(np.eye(2)[meta.Org[valid_i]],axis=1),maxlen,axis=1)
    #Random annotations are added as input
    x_valid_target_inp = np.random.randint(1,size=(len(valid_i),maxlen))
    x_valid = [x_valid_seqs,x_valid_target_inp,x_valid_orgs] #inp seq, target annoation, organism
    y_valid = [CS[valid_i],Types[valid_i]]

    #Model
    #Based on: https://keras.io/examples/nlp/text_classification_with_transformer/
    #Variable params
    embed_dim = int(net_params['embed_dim']) #32  # Embedding size for each token
    num_heads = int(net_params['num_heads']) #1  # Number of attention heads
    ff_dim = int(net_params['ff_dim']) #32  # Hidden layer size in feed forward network inside transformer
    num_layers = int(net_params['num_layers']) #1  # Number of attention heads
    batch_size = int(net_params['batch_size']) #32

    #Create model
    model = create_model(maxlen, vocab_size, embed_dim,num_heads, ff_dim,num_layers, find_lr)

    #Summary of model
    #print(model.summary())


    if find_lr == True:
        lr_finder = LRFinder(model)
        lr_finder.find(x_train, y_train, start_lr=0.00001, end_lr=1, batch_size=batch_size, epochs=1)
        losses = lr_finder.losses
        lrs = lr_finder.lrs
        l_l = np.asarray([lrs, losses])
        np.savetxt(outdir+'lrs_losses'+str(param_combo)+'.txt', l_l)
        num_epochs = 0
        break


    else:
        #Checkpoint
        if checkpoint == True:
            #Make dir
            try:
                os.mkdir(checkpointdir+'TP'+str(test_partition)+'/VP'+str(valid_partition))
            except:
                print('Checkpoint directory exists...')

            checkpoint_path=checkpointdir+'TP'+str(test_partition)+'/VP'+str(valid_partition)+"/weights_"+str(param_combo)+"_{epoch:02d}.hdf5"
            checkpointer = tf.keras.callbacks.ModelCheckpoint(
            checkpoint_path, monitor='val_loss', verbose=0, save_best_only=True,
            save_weights_only=True, mode='auto', save_freq='epoch')

            #Callbacks
            callbacks=[checkpointer]
        else:
            callbacks = []

        history = model.fit(
            x_train, y_train, batch_size=batch_size, epochs=num_epochs,
            validation_data=(x_valid, y_valid),
            callbacks=callbacks
        )

        #Save loss
        train_losses.append(history.history['loss'])
        valid_losses.append(history.history['val_loss'])

if find_lr != True:
    #Save array of losses
    outid = str(test_partition)+'_'+str(param_combo)
    np.save(outdir+'train_losses_'+outid+'.npy',np.array(train_losses))
    np.save(outdir+'valid_losses_'+outid+'.npy',np.array(valid_losses))
    print('Done')
