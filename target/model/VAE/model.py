
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
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

class Sampling(layers.Layer):
    """Uses (z_mean, z_log_var) to sample z, the vector encoding a digit."""

    def call(self, inputs):
        z_mean, z_log_var = inputs
        batch = tf.shape(z_mean)[0]
        dim = tf.shape(z_mean)[1]
        epsilon = tf.keras.backend.random_normal(shape=(batch, dim))

        return z_mean + tf.exp(0.5 * z_log_var) * epsilon

def create_model(maxlen, vocab_size, embed_dim,num_heads, ff_dim,num_layers, find_lr):
    '''Create the transformer model
    '''

    #Latent dim - dimension for sampling
    latent_dim = int(ff_dim/8)
    #Encoder
    encoder_inp = layers.Input(shape=(maxlen)) #Input AA sequence
    ##Embeddings
    embedding_layer = TokenAndPositionEmbedding(maxlen, vocab_size, embed_dim)
    encoder = EncoderBlock(embed_dim, num_heads, ff_dim)

    #Encode
    x = embedding_layer1(encoder_input)
    x, enc_attn_weights = encoder(x,x,x)

    #Constrain to distribution
    z_mean = layers.Dense(latent_dim, name="z_mean")(x)
    z_log_var = layers.Dense(latent_dim, name="z_log_var")(x)
    #z = z_mean + exp(z_log_sigma) * epsilon, where epsilon is a random normal tensor.
    z = Sampling()([z_mean, z_log_var])
    #model
    encoder = keras.Model(encoder_inp, [z_mean, z_log_var, z], name="encoder")
    print(encoder.summary())

    #Decoder
    latent_inp = keras.Input(shape=(latent_dim,))

    #Final
    preds = layers.Dense(vocab_size, activation="softmax")(x) #Annotate
    #model
    decoder = keras.Model(latent_inp, preds, name="decoder")
    print(decoder.summary())

    #VAE
    vae_outp = decoder(encoder(encoder_inp)[2]) #Inp z to decoder
    vae = keras.Model(encoder_inp, vae_outp, name='vae')
    #Loss
    reconstruction_loss = keras.losses.mean_absolute_error(encoder_inp,vae_outp)
    #reconstruction_loss *= 10
    kl_loss = 1 + z_log_var - keras.backend.square(z_mean) - keras.backend.exp(z_log_var)
    kl_loss = keras.backend.sum(kl_loss, axis=-1)
    kl_loss *= -0.5
    vae_loss = keras.backend.mean(reconstruction_loss + kl_loss)

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
