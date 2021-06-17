
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from attention_class import MultiHeadSelfAttention
import pdb
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


def create_model(maxlen, vocab_size, embed_dim,num_heads, encode_dim,num_layers, find_lr):
    '''Create the transformer model
    '''

    #Latent dim - dimension for sampling
    latent_dim = encode_dim
    #Encoder
    encoder_inp = layers.Input(shape=(maxlen)) #Input AA sequence
    ##Embeddings
    embedding_layer = TokenAndPositionEmbedding(maxlen, vocab_size, embed_dim)
    #enc_attention = MultiHeadSelfAttention(embed_dim,num_heads)
    enc_attention = EncoderBlock(embed_dim, num_heads, encode_dim)
    #Encode
    x = embedding_layer(encoder_inp)
    #Attention layer
    for i in range(num_layers):
        x, enc_attn_weights = enc_attention(x,x,x)
    #Flatten
    x = layers.Flatten()(x) # (batch_size, seq_len, encode_dim)
    z = layers.Dense(latent_dim, name="z")(x)
    #model
    encoder = keras.Model(encoder_inp, [z], name="encoder")
    print(encoder.summary())

    #Decoder
    latent_inp = keras.Input(shape=(latent_dim))
    #Dense
    x = layers.Dense(maxlen*embed_dim,activation='relu')(latent_inp)
    #Reshape
    x = layers.Reshape((maxlen,embed_dim))(x)
    #decoder attention
    #dec_attention = MultiHeadSelfAttention(embed_dim,num_heads)
    dec_attention = EncoderBlock(embed_dim, num_heads, encode_dim)
    for i in range(num_layers):
        x, dec_attn_weights = dec_attention(x,x,x)
    #Final
    preds = layers.Dense((vocab_size), activation="softmax")(x) #Annotate
    #model
    decoder = keras.Model(latent_inp, preds, name="decoder")
    print(decoder.summary())

    #VAE
    vae_outp = decoder(encoder(encoder_inp)) #Inp z to decoder
    vae = keras.Model(encoder_inp, vae_outp, name='vae')
    #Loss
    vae_loss = keras.losses.SparseCategoricalCrossentropy()(encoder_inp,vae_outp) #true,pred

    #learning_rate
    initial_learning_rate = 1e-3 #From lr finder

    if find_lr ==False:
        lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate,
        decay_steps=10000,
        decay_rate=0.96,
        staircase=True)
    else:
        lr_schedule = initial_learning_rate

    opt = tf.keras.optimizers.Adam(learning_rate = lr_schedule,amsgrad=True,clipnorm=1.0)

    #Compile
    vae.add_loss(vae_loss)
    vae.compile(optimizer=opt,metrics=["accuracy"])

    return vae
