

import argparse
import sys
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


#visualization
from tensorflow.keras.callbacks import TensorBoard

from transformer_classes import MultiHeadAttention, Encoder, Decoder
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
class Transformer(tf.keras.Model):
  def __init__(self, num_layers, d_model, num_heads, dff, input_vocab_size,
               target_vocab_size, pe_input, pe_target, rate=0.1):
    super(Transformer, self).__init__()

    self.tokenizer = Encoder(num_layers, d_model, num_heads, dff,
                           input_vocab_size, pe_input, rate)

    self.decoder = Decoder(num_layers, d_model, num_heads, dff,
                           target_vocab_size, pe_target, rate)

    self.final_layer = tf.keras.layers.Dense(target_vocab_size)

  def call(self, inp, tar, training, enc_padding_mask,
           look_ahead_mask, dec_padding_mask):

    enc_output = self.tokenizer(inp, training, enc_padding_mask)  # (batch_size, inp_seq_len, d_model)

    # dec_output.shape == (batch_size, tar_seq_len, d_model)
    dec_output, attention_weights = self.decoder(
        tar, enc_output, training, look_ahead_mask, dec_padding_mask)

    final_output = self.final_layer(dec_output)  # (batch_size, tar_seq_len, target_vocab_size)

    return final_output, attention_weights

###MASKING
def create_padding_mask(seq):
  seq = tf.cast(tf.math.equal(seq, 0), tf.float32)

  # add extra dimensions to add the padding
  # to the attention logits.
  return seq[:, tf.newaxis, tf.newaxis, :]  # (batch_size, 1, 1, seq_len)

def create_look_ahead_mask(size):
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

def create_model(maxlen, vocab_size, embed_dim,num_heads, ff_dim,num_layers):
    '''Create the transformer model
    '''

    seq_input = layers.Input(shape=(maxlen,)) #Input aa sequences
    seq_target = layers.Input(shape=(maxlen,)) #Targets - annotations
    kingdom_input = layers.Input(shape=(4,)) #4 kingdoms, Archaea, Eukarya, Gram +, Gram -

    #Define the transformer
    transformer = Transformer(num_layers, embed_dim, num_heads, ff_dim, 21, 6, 70,70)
    enc_padding_mask, combined_mask, dec_padding_mask = create_masks(inp, tar_inp)
    x = tranformer(seq_input,seq_target,
                    True,
                    enc_padding_mask, combined_mask, dec_padding_mask)


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


    model = keras.Model(inputs=[seq_input,seq_target,kingdom_input], outputs=[preds,pred_type])
    #Optimizer
    opt = keras.optimizers.Adam(learning_rate=0.001,amsgrad=True)
    #Compile
    model.compile(optimizer = opt, loss= [SparseCategoricalFocalLoss(gamma=2),SparseCategoricalFocalLoss(gamma=2)], metrics=["accuracy"])

    return model

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
    print('Validation partition',valid_partition)
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

    #Create model
    model = create_model(maxlen, vocab_size, embed_dim,num_heads, ff_dim,num_layers)
    pdb.set_trace()
    #Save model
    if save_model == True:
        model_json = model.to_json()
        with open(checkpointdir+"model.json", "w") as json_file:
      	     json_file.write(model_json)

    #Summary of model
    #print(model.summary())
    #Checkpoint
    if checkpoint == True:
        #Make dir
        try:
            os.mkdir(checkpointdir+'vp'+str(valid_partition))
        except:
            print('Checkpoint directory exists...')
        checkpoint_path=checkpointdir+'vp'+str(valid_partition)+"/weights_{epoch:02d}.hdf5"
        model_checkpoint = ModelCheckpoint(checkpoint_path, verbose=0, monitor="val_loss",save_best_only=True, mode='min',overwrite=False)
        #Callbacks
        callbacks=[model_checkpoint]
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
