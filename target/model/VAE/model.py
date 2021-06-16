
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
#####FUNCTIONS and CLASSES#####
class Sampling(layers.Layer):
    """Uses (z_mean, z_log_var) to sample z, the vector encoding a digit."""

    def call(self, inputs):
        z_mean, z_log_var = inputs
        batch = tf.shape(z_mean)[0]
        dim = tf.shape(z_mean)[1]
        epsilon = tf.keras.backend.random_normal(shape=(batch, dim))

        return z_mean + tf.exp(0.5 * z_log_var) * epsilon

def create_model(maxlen, encode_dim):
    '''Create the transformer model
    '''

    #Latent dim - dimension for sampling
    latent_dim = int(encode_dim/8)
    #Encoder
    encoder_inp = layers.Input(shape=(maxlen)) #Input methylation profiles
    x = layers.Dense(encode_dim,activation='relu')(encoder_inp)
    x = layers.Dropout(0.1)(x)
    x = layers.BatchNormalization()(x) #Bacth normalize, focus on segment
    x = layers.Dense(int(encode_dim/2),activation='relu')(x)
    x = layers.Dropout(0.1)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dense(int(encode_dim/4),activation='relu')(x)
    x = layers.Dropout(0.1)(x)
    x = layers.BatchNormalization()(x)
    #Constrain to distribution
    z_mean = layers.Dense(latent_dim, name="z_mean")(x)
    z_log_var = layers.Dense(latent_dim, name="z_log_var")(x)
    #z = z_mean + exp(z_log_sigma) * epsilon, where epsilon is a random normal tensor.
    z = Sampling()([z_mean, z_log_var])
    #model
    encoder = keras.Model(encoder_inp, [z_mean, z_log_var, z], name="encoder")
    #encoder.summary()

    #Decoder
    latent_inp = keras.Input(shape=(latent_dim,))
    x = layers.Dense(int(encode_dim/4),activation='relu')(latent_inp)
    x = layers.Dropout(0.1)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dense(int(encode_dim/2),activation='relu')(x)
    x = layers.Dropout(0.1)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dense(encode_dim,activation='relu')(x)
    x = layers.Dropout(0.1)(x)
    x = layers.BatchNormalization()(x)
    #Final
    preds = layers.Dense(maxlen, activation="sigmoid")(x) #Annotate
    #model
    decoder = keras.Model(latent_inp, preds, name="decoder")
    #decoder.summary()

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
