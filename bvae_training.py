import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["KERAS BACKEND"] = "tensorflow"
import pandas as pd
import numpy as np
import tensorflow as tf
import keras
from keras import layers
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from spectrum_preprocessing import roundWavenumbers, distribution_Selection, pipeline
import matplotlib.pyplot as plt

"""
Linear Callback for Annealing the Beta Value

"""
class LinearBetaAnneal(keras.callbacks.Callback):
    """
    Anneals the beta variable in betaVAE in a linear motion:
    beta begins at zero, then has a warmup period given by the programmer, 
    then once it reaches the maximum allowed beta value, maintains its value
    for the remainder of the training epochs. 
    
    """
    def __init__(self, vae, warmup_epochs=10, beta_max=15):
        self.vae = vae
        self.warmup_epochs = warmup_epochs
        self.beta_max = beta_max

    def on_epoch_begin(self, epoch, logs=None):
        vae.beta.assign(self.beta_max * min(1.0 , epoch / self.warmup_epochs))


"""
Cyclical Callback for Annealing the Beta Value

"""
class CyclicalBetaAnneal(keras.callbacks.Callback):
    """
    Anneals the beta variable in betaVAE in a sawtooth motion: 
    beta begins at zero, then has a warmup period of (by default) half the cycle length,
    holds the maximum beta for (by default) half of a cycle, and then returns beta to zero to repeat 
    the process as many times as possible for given epochs.  
    
    """
    def __init__(self, vae, cycle_length=20, warmup_ratio=0.5, beta_max=15):
        self.vae = vae
        self.cycle_length = cycle_length
        self.warmup_ratio = warmup_ratio
        self.beta_max = beta_max

    def on_epoch_begin(self, epoch, logs=None):
        cycle_pos = epoch % self.cycle_length
        warmup_epochs = int(self.cycle_length * self.warmup_ratio)
        vae.beta.assign(self.beta_max * min(1.0, cycle_pos / warmup_epochs))
        

"""
Reparameterization 

"""
class Sampling(layers.Layer):
    def call(self, inputs, training=None):
        z_mean, z_logvar = inputs
        if training:
            eps = tf.random.normal(shape=tf.shape(z_mean))
            return z_mean + tf.exp(0.5 * z_logvar) * eps
        return z_mean 

"""
BetaVAE Model

"""

class BetaVAE(keras.Model):
    def __init__(self, encoder, decoder, beta=1.0, **kwargs):
        super().__init__(**kwargs)
        self.encoder = encoder
        self.decoder = decoder
        self.beta = tf.Variable(beta, trainable=False, dtype=tf.float32)
        # Trackers for clean progress bars
        self.total_loss_tracker = keras.metrics.Mean(name="total_loss")
        self.recon_loss_tracker = keras.metrics.Mean(name="recon_loss")
        self.kl_loss_tracker = keras.metrics.Mean(name="kl_loss")

    @property
    def metrics(self):
        return [self.total_loss_tracker, self.recon_loss_tracker, self.kl_loss_tracker]

    def train_step(self, data):
        # Data is usually (x, y); for VAE, we use x for both
        if isinstance(data, tuple):
            data = data[0]
            
        with tf.GradientTape() as tape:
            z_mean, z_logvar, z = self.encoder(data, training=True)
            reconstruction = self.decoder(z, training=True)
            
            # Reconstruction: MSE summed over features, then averaged over batch
            recon_loss = tf.math.reduce_mean(
                tf.math.reduce_sum(tf.math.square(data - reconstruction), axis=1)
            )
            
            # KL Divergence: Summed over latent dims, then averaged over batch
            kl_loss = -0.5 * tf.math.reduce_sum(
                1 + z_logvar - tf.math.square(z_mean) - tf.math.exp(z_logvar), axis=1
            )
            kl_loss = tf.math.reduce_mean(kl_loss)
            
            total_loss = recon_loss + self.beta * kl_loss

        grads = tape.gradient(total_loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
        
        self.total_loss_tracker.update_state(total_loss)
        self.recon_loss_tracker.update_state(recon_loss)
        self.kl_loss_tracker.update_state(kl_loss)
        return {m.name: m.result() for m in self.metrics}

    def test_step(self, data):
        if isinstance(data, tuple):
            data = data[0]
        
        z_mean, z_logvar, z = self.encoder(data, training=False)
        reconstruction = self.decoder(z, training=False)
        
        recon_loss = tf.math.reduce_mean(tf.reduce_sum(tf.square(data - reconstruction), axis=1))
        kl_loss = -0.5 * tf.math.reduce_sum(1 + z_logvar - tf.math.square(z_mean) - tf.math.exp(z_logvar), axis=1)
        kl_loss = tf.math.reduce_mean(kl_loss)
        
        total_loss = recon_loss + self.beta * kl_loss
        
        self.total_loss_tracker.update_state(total_loss)
        self.recon_loss_tracker.update_state(recon_loss)
        self.kl_loss_tracker.update_state(kl_loss)
        return {m.name: m.result() for m in self.metrics}
    
    def get_config(self):
        # Return serializable config
        config = super().get_config()
        config.update({
            "units": self.units,
            "activation": tf.keras.activations.serialize(self.activation)
        })
        return config

    def call(self, x):
        # Standard call returns the reconstruction
        _, _, z = self.encoder(x)
        return self.decoder(z)


"""
Prepare the training and validation data

"""
training_df = pd.read_csv("interpolated_training_data.csv", low_memory=False)
validation_df = pd.read_csv("interpolated_testing_data.csv", low_memory=False)

frequencies = training_df.columns.astype(float)
wavenumbers = training_df.columns.astype(str)

betaVAE_trainingData = training_df[wavenumbers]
#betaVAE_validationData = validation_df[wavenumbers]

trainingArray = np.asarray(betaVAE_trainingData.values, dtype=np.float32)
#validationArray = np.asarray(betaVAE_validationData.values, dtype=np.float32)

X_train, X_val = train_test_split(trainingArray, train_size=0.9, test_size=0.1, shuffle=True)

print(X_train.shape, X_val.shape)
training_df = pd.DataFrame(data=X_train, columns=wavenumbers)
training_df.to_csv("training_data.csv", index=False)
validation_df = pd.DataFrame(data=X_val, columns=wavenumbers)
validation_df.to_csv("validation_data.csv", index=False)

training_df = None # Clear memory
validation_df = None # Clear memory
betaVAE_trainingData = None # Clear memory
betaVAE_validationData = None # Clear memory

"""
Define the model parameters

"""
input_dim = len(wavenumbers)
output_dim = input_dim

batch = 32

hidden_dims = [512, 256, 128]

latent_dim = 16

beta = 2

epochs = 200

"""
Build the Encoder

"""
def make_encoder(input_dim, latent_dim, hidden):
    x_in = keras.Input(shape=(input_dim,), name="x")
    x=x_in
    x = layers.GaussianNoise(0.05)(x)
    for i, h in enumerate(hidden):
        x = layers.Dense(h, activation="gelu", name=f"enc_dense_{i}")(x)
    z_mean   = layers.Dense(latent_dim, name="z_mean")(x)
    z_logvar = layers.Dense(latent_dim, name="z_logvar")(x)
    z = Sampling()([z_mean, z_logvar])
    return keras.Model(x_in, [z_mean, z_logvar, z], name="encoder")

"""
Build the Decoder 

"""
def make_decoder(output_dim, latent_dim, hidden):
    z_in = keras.Input(shape=(latent_dim,), name="z")
    x = z_in
    for i, h in enumerate(hidden[::-1]):
        x = layers.Dense(h, activation="gelu", name=f"dec_dense_{i}")(x)
    x_out = layers.Dense(output_dim, activation="linear", name="x_recon")(x)
    return keras.Model(z_in, x_out, name="decoder")

encoder = make_encoder(input_dim, latent_dim, hidden_dims)
decoder = make_decoder(input_dim, latent_dim, hidden_dims)

"""
Build the VAE Model and Train

"""
#vae = BetaVAE(encoder, decoder, beta, 100, c_steps=1e5)
vae = BetaVAE(encoder, decoder, beta)

vae.compile(
    optimizer=keras.optimizers.Adam(learning_rate=1e-4)
)

callbacks = [
    keras.callbacks.EarlyStopping(monitor='val_total_loss', patience=12, restore_best_weights=True),
    #keras.callbacks.ReduceLROnPlateau(monitor="val_total_loss", factor=0.1, patience=10, min_lr=1e-5),
    keras.callbacks.TerminateOnNaN(),
    #LinearBetaAnneal(vae, warmup_epochs=10, beta_max=beta)
    ]

history = vae.fit(X_train, validation_data=(X_val,), epochs=epochs, batch_size=batch, callbacks=callbacks, verbose=1)

"""
Print the KL divergence for each latent dimension on the validation data

"""
z_mean, z_logvar, _ = encoder(X_train, training=False)
print("Z Mean shape:", z_mean.shape)
print("Z Log Var shape:", z_logvar.shape)

"""
Save the model and reload to test that it saved correctly

"""

tf.saved_model.save(encoder, './new_encoder/')
tf.saved_model.save(decoder, './new_decoder/')
tf.saved_model.save(vae, './new_vae/')

encoder = None # Clear memory
decoder = None # Clear memory

saved_encoder = tf.saved_model.load("./new_encoder/")
saved_decoder = tf.saved_model.load("./new_decoder/")
saved_vae = tf.saved_model.load("./new_vae/")

if saved_encoder is not None and saved_decoder is not None and saved_vae is not None:
    print("All models loaded successfully!")

else:
    print("Failed to load the model.")


