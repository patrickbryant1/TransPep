

import argparse
import sys
import numpy as np
import pandas as pd
import time
from collections import Counter
#Preprocessing and evaluation
from process_data import parse_and_format, eval_cs

#Keras
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from categorical_focal_loss import SparseCategoricalFocalLoss
from tensorflow.keras.callbacks import ModelCheckpoint


#visualization
from tensorflow.keras.callbacks import TensorBoard

from multi_head_attention import MultiHeadSelfAttention
#from lr_finder import LRFinder


import pdb
#Arguments for argparse module:
parser = argparse.ArgumentParser(description = '''A Transformer Neural Network for analyzing signal peptides.''')

parser.add_argument('--train_data', nargs=1, type= str, default=sys.stdin, help = 'Path to training data in fasta format.')
parser.add_argument('--datadir', nargs=1, type= str, default=sys.stdin, help = 'Path to data directory.')
parser.add_argument('--test_partition', nargs=1, type= int, default=sys.stdin, help = 'Which CV fold to test on.')
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
class TransformerBlock(layers.Layer):
    def __init__(self, embed_dim, num_heads, ff_dim, rate=0.1):
        super(TransformerBlock, self).__init__()
        self.att = MultiHeadSelfAttention(embed_dim,num_heads)
        self.ffn = keras.Sequential(
            [layers.Dense(ff_dim, activation="relu"), layers.Dense(embed_dim),]
        )
        self.layernorm1 = layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = layers.Dropout(rate)
        self.dropout2 = layers.Dropout(rate)

    def call(self, inputs, training):
        attn_output = self.att(inputs)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(inputs + attn_output)
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        return self.layernorm2(out1 + ffn_output)

    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'embed_dim': embed_dim,
            'num_heads': num_heads,
            'ff_dim': ff_dim
        })
        return config

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

    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'maxlen': maxlen,
            'vocab_size': vocab_size,
            'embed_dim': embed_dim
        })
        return config

def construct_train_valid_data():
    '''Construct the train and validation data
    '''

######################MAIN######################
args = parser.parse_args()
datadir = args.datadir[0]
try:
    train_meta = pd.read_csv(datadir+'train_meta.csv')
    train_seqs = np.load(datadir+'seqs.npy',allow_pickle=True)
    train_annotations = np.load(datadir+'annotations.npy',allow_pickle=True)
except:
    train_meta, train_seqs, train_annotations = parse_and_format(args.train_data[0])
    #Save
    train_meta.to_csv(datadir+'train_meta.csv')
    np.save(datadir+'seqs.npy',train_seqs)
    np.save(datadir+'annotations.npy',train_annotations)

#Get parameters
variable_params=pd.read_csv(args.variable_params[0])
param_combo=args.param_combo[0]
test_partition = args.test_partition[0]
checkpointdir = args.checkpointdir[0]
save_model = bool(args.save_model[0])
checkpoint = bool(args.checkpoint[0])
num_epochs = args.num_epochs[0]
outdir = args.outdir[0]
pdb.set_trace()
train_CS = train_meta.CS.values
train_kingdoms = train_meta.Kingdom.values
train_meta['Type'] = train_meta['Type'].replace({'NO_SP':0,'SP':1,'TAT':2,'LIPO':3})
train_types = train_meta.Type.values
#Onehot conversion
train_kingdoms = np.eye(4)[train_kingdoms]

#Get data
#Run through all by taking as input
test_i = train_meta[train_meta.Partition==test_partition].index
train_losses = []
valid_losses = []
for valid_partition in np.setdiff1d(np.arange(5),test_partition):
    valid_i = train_meta[train_meta.Partition==valid_partition].index
    train_i = np.setdiff1d(np.arange(len(train_meta)),np.concatenate([test_i,valid_i]))
    #train
    x_train_seqs = train_seqs[train_i]
    x_train_kingdoms = train_kingdoms[train_i]
    x_train = [x_train_seqs,x_train_kingdoms]
    y_train = [train_annotations[train_i],train_types[train_i]]
    #valid
    x_valid_seqs = train_seqs[valid_i]
    x_valid_kingdoms = train_kingdoms[valid_i]
    x_valid = [x_valid_seqs,x_valid_kingdoms]
    y_valid = [train_annotations[valid_i],train_types[valid_i]]
    #Construct weights
    #y_flat = y_train[0].flatten()
    #counts = Counter(y_flat)
    #class_weights = {}
    #for key in counts:
    #    class_weights[key] = counts[key]/len(y_flat)

    #Model
    #Based on: https://keras.io/examples/nlp/text_classification_with_transformer/
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

    seq_input = layers.Input(shape=(maxlen,))
    kingdom_input = layers.Input(shape=(4,)) #4 kingdoms, Archaea, Eukarya, Gram +, Gram -
    embedding_layer = TokenAndPositionEmbedding(maxlen, vocab_size, embed_dim)
    x = embedding_layer(seq_input)
    transformer_block = TransformerBlock(embed_dim, num_heads, ff_dim)

    #Stacking transformer blocks
    for ti in range(num_layers):
        x = transformer_block(x) #Can add more transformer blocks here

    x = layers.GlobalAveragePooling1D()(x)
    x = layers.Dropout(0.1)(x)
    x = layers.Dense(20, activation="relu")(x)
    x = layers.Dropout(0.1)(x)
    #Concat
    x = layers.Concatenate()([x,kingdom_input])
    preds = layers.Dense(70*6, activation="softmax")(x)
    pred_type = layers.Dense(4, activation="softmax",name='type')(x) #Type of protein
    preds = layers.Reshape((-1,70,6), name='annotation')(preds)
    #pred_cs = layers.Dense(1, activation="elu", name='pred_cs')(x)


    model = keras.Model(inputs=[seq_input,kingdom_input], outputs=[preds,pred_type])
    #Optimizer
    opt = keras.optimizers.Adam(learning_rate=0.001,amsgrad=True)
    #Compile
    model.compile(optimizer = opt, loss= [SparseCategoricalFocalLoss(gamma=2),SparseCategoricalFocalLoss(gamma=2)], metrics=["accuracy"])

    #Save model
    if save_model == True:
        model_json = model.to_json()
        with open(checkpointdir+"model.json", "w") as json_file:
      	     json_file.write(model_json)

    #Summary of model
    #print(model.summary())
    #Checkpoint
    if checkpoint == True:
        checkpoint_path=checkpointdir+"weights_{epoch:02d}_vp"+str(valid_partition)+"_.hdf5"
        checkpoint = ModelCheckpoint(checkpoint_path, verbose=0, monitor="val_loss",save_best_only=True, mode='min',overwrite=False)
        #Callbacks
        callbacks=[checkpoint]
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

#Save array of losses
outid = str(test_partition)+'_'+str(param_combo)
np.save(outdir+'train_losses_'+outid+'.npy',np.array(train_losses))
np.save(outdir+'valid_losses_'+outid+'.npy',np.array(valid_losses))
print('Done')
