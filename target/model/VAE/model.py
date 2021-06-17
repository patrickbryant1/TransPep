
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from attention_class import MultiHeadSelfAttention
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

class Sampling(layers.Layer):
    """Uses (z_mean, z_log_var) to sample z, the vector encoding a digit."""

    def call(self, inputs):
        z_mean, z_log_var = inputs
        batch = tf.shape(z_mean)[0]
        dim = tf.shape(z_mean)[1]
        epsilon = tf.keras.backend.random_normal(shape=(batch, dim))

        return z_mean + tf.exp(0.5 * z_log_var) * epsilon

def create_model(maxlen, vocab_size, embed_dim,num_heads, encode_dim,num_layers, find_lr):
    '''Create the transformer model
    '''

    #Latent dim - dimension for sampling
    latent_dim = encode_dim
    #Encoder
    encoder_inp = layers.Input(shape=(maxlen)) #Input AA sequence
    ##Embeddings
    embedding_layer = TokenAndPositionEmbedding(maxlen, vocab_size, embed_dim)
    enc_attention = MultiHeadSelfAttention(embed_dim,num_heads)

    #Encode
    x = embedding_layer(encoder_inp)
    #Attention layer
    for i in range(2):
        x, enc_attn_weights = enc_attention(x,x,x)
    #Flatten
    x = layers.Reshape((maxlen*embed_dim,))(x) # (batch_size, seq_len, embed_dim)
    z = layers.Dense(latent_dim, name="z")(x)
    # #Constrain to distribution
    # z_mean = layers.Dense(latent_dim, name="z_mean")(x)
    # z_log_var = layers.Dense(latent_dim, name="z_log_var")(x)
    # #z = z_mean + exp(z_log_sigma) * epsilon, where epsilon is a random normal tensor.
    # z = Sampling(name='z')([z_mean, z_log_var])
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
    dec_attention = MultiHeadSelfAttention(embed_dim,num_heads)
    for i in range(2):
        x, enc_attn_weights = dec_attention(x,x,x)
    #Final
    preds = layers.Dense((vocab_size), activation="softmax")(x) #Annotate
    #model
    decoder = keras.Model(latent_inp, preds, name="decoder")
    print(decoder.summary())

    #VAE
    vae_outp = decoder(encoder(encoder_inp)) #Inp z to decoder
    vae = keras.Model(encoder_inp, vae_outp, name='vae')
    #Loss
    reconstruction_loss = keras.losses.SparseCategoricalCrossentropy()(encoder_inp,vae_outp) #true,pred
    #kl loss
    # kl_loss = 1 + z_log_var - keras.backend.square(z_mean) - keras.backend.exp(z_log_var)
    # kl_loss = keras.backend.sum(kl_loss, axis=-1)
    # kl_loss *= -0.5
    vae_loss = reconstruction_loss  #keras.backend.mean(reconstruction_loss + kl_loss)

    #Optimizer
    initial_learning_rate = 1e-3
    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate,
    decay_steps=10000,
    decay_rate=0.96,
    staircase=True)

    opt = tf.keras.optimizers.Adam(learning_rate = 1e-3,amsgrad=True,clipnorm=1.0)

    #Compile
    vae.add_loss(vae_loss)
    vae.compile(optimizer=opt,metrics=["accuracy"])

    return vae
