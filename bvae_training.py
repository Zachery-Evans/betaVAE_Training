import os
import re
import pandas as pd
import numpy as np
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
#os.environ["TF_USE_LEGACY_KERAS"] = "1"
import tensorflow as tf
import keras
from keras import Model
from keras import layers
from scipy.stats import gaussian_kde
from sklearn.model_selection import cross_val_score, KFold
import random
from spectrum_preprocessing import interpolate_spectrum

def roundWavenumbers(dataframe):
    """
    Function for rounding the wavenumbers so that there is no mismatch between the machine 
    precision of saving the files in Quasar and the wavenumbers output by the FTIR microscope
    """
    last_nonwavenum_idx = dataframe.columns.get_loc('1981.7 - 2095.8') + 1

    dataframe = dataframe.rename(columns=lambda c: round(float(c), 1) if c not in dataframe.columns[:last_nonwavenum_idx] else c)
    return dataframe

def distribution_Selection(df, distributionIdx, numberOfSigmas):
    """
    Screens data based on the distribution of a given column in a pandas dataframe. 
    Built for screening out dataset values that do not have a sufficient 2019 wavenumber 
    integral for the PEX project. Can be modified to projects liking. 

    Returns the selected datas indexes, the discarded indexes, the mask used to select the data,
    the x position of the distribution mode, and the column values of the distribution that is
    being used to screen data. In this case the values of the baseline integral of the PE 2019
    wavenumber peak. 
    """
    
    df[distributionIdx] = df[distributionIdx].astype(float)

    #creating variable for the array of Polyethylene Area
    area = df['1981.7 - 2095.8'].values

    # Use Gaussian KDE to find the center position of the rightmost mode by creating x coords
    # and then using np.argmax to determine the point with the highest number of counts
    kde = gaussian_kde(area)
    xs = np.linspace(area.min(), area.max(), 1000) 
    modePosition = xs[np.argmax(kde(xs))]

    # Use only the values greater than 1.5 to determine the std of the PE peaks
    positive_vals = area[area > 1.5]
    sigma = positive_vals.std()

    lowerDistributionBound = modePosition - numberOfSigmas * sigma
    upperDistributionBound = modePosition + numberOfSigmas * sigma

    # Select the range of PE normalization integral to be accepted by the mask
    mask_selected = (area >= lowerDistributionBound) & (area <= upperDistributionBound)
    selected_indexes = df.index[mask_selected]
    discarded_indexes = df.index[~mask_selected]

    return selected_indexes, discarded_indexes, mask_selected, modePosition, area

"""
Build the Encoder
"""
def build_encoder(input_dim, latent_dim): 

    inputs = keras.Input(shape=input_dim) 
    x = layers.Dense(128, activation="relu")(inputs) 
    x = layers.Dense(128, activation='relu')(x) 
    z_mean = layers.Dense(latent_dim, name="z_mean")(x) 
    z_logvar = layers.Dense(latent_dim, name="z_logvar")(x) 

    return keras.Model(inputs, [z_mean, z_logvar], name="encoder")

"""
Build the Decoder 
"""
def build_decoder(latent_dim, output_dim):
    latent_inputs = keras.Input(shape=latent_dim)
    x = layers.Dense(128, activation="relu")(latent_inputs)
    x = layers.Dense(128, activation='relu')(x)
    outputs = layers.Dense(output_dim, activation="linear")(x)
    return keras.Model(latent_inputs, outputs, name="decoder")


"""
Reparameterization 

"""
class Sampling(keras.layers.Layer):
    def call(self, inputs):
        z_mean, z_log_var = inputs
        epsilon = tf.random.normal(shape=tf.shape(z_mean))
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon

"""
Subclass of BetaVAE

"""
class BetaVAE(keras.Model):
    def __init__(self, encoder, decoder, beta, **kwargs):
        super().__init__(**kwargs)
        self.encoder = encoder
        self.decoder = decoder
        self.beta = beta
        self.sampling = Sampling()

        self.total_loss_tracker = keras.metrics.Mean(name="loss")
        self.recon_loss_tracker = keras.metrics.Mean(name="recon_loss")
        self.kl_loss_tracker = keras.metrics.Mean(name="kl_loss")

    def call(self, inputs, training=False):
        z_mean, z_log_var = self.encoder(inputs)
        z = self.sampling([z_mean, z_log_var])
        return self.decoder(z)

    @property
    def metrics(self):
        return [
            self.total_loss_tracker,
            self.recon_loss_tracker,
            self.kl_loss_tracker,
        ]
    
    def get_config(self):
        config = super().get_config()
        config.update({
            "beta": self.beta,
        })
        return config
    
    @classmethod
    def from_config(cls, config):
        # You MUST recreate encoder and decoder here
        encoder = build_encoder(input_dim=505, latent_dim=16)
        decoder = build_decoder(latent_dim=16, output_dim=505)
        return cls(encoder=encoder, decoder=decoder, **config)
    
    """
    The training of the Model, here we have the minimization of the loss function and define the loss
    function of the beta VAE to include the beta normalization term.
    """
    def train_step(self, data):
        
        if isinstance(data, tuple):
            data = data[0]

        with tf.GradientTape() as tape:
            reconstruction = self(data, training=True) 

            z_mean, z_logvar = self.encoder(data, training=True)
            
            recon_loss = tf.reduce_mean(
                tf.reduce_sum(
                    keras.losses.binary_crossentropy(data, reconstruction),
                )
            )

            kl_loss = -0.5 * tf.reduce_mean(
                tf.reduce_sum(1 + z_logvar - tf.square(z_mean) - tf.exp(z_logvar), axis=1)
            )

            total_loss = recon_loss + self.beta * kl_loss

        grads = tape.gradient(total_loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))

        return {
            "loss": total_loss,
            "reconstruction_loss": recon_loss,
            "kl_loss": kl_loss,
        }



"""
Load and Format the data for k-fold validation of the beta-VAE model
"""

path = './training_data/'
#List all of the files in the data directory.
allFiles = os.listdir(path)
# Take only the files that contain data pertaining to SMP65#010 
trainingFiles = [file for file in allFiles if file.endswith('.csv') and 'SMP65' in file and 'full width' not in file]

trainingFiles = sorted(trainingFiles, key=lambda x: int(re.search(r'(?<= )(.+?)(?=d)', x).group()))

# Read all of the data and place the dataframes into a list
trainingDataframeList = [pd.read_csv(path+file) for file in trainingFiles]

test_df = trainingDataframeList[0]
test_df = roundWavenumbers(test_df)

last_nonwavenum_idx = test_df.columns.get_loc('1981.7 - 2095.8') + 1
wavenumbers = test_df.columns[last_nonwavenum_idx:]

masked_trainingDataframeList = []

stdDevs = 3

for df in trainingDataframeList:
    selected_indexes, discarded_indexes, mask_selected, modePosition, areaPE = distribution_Selection(df, '1981.7 - 2095.8', stdDevs)
    masked_trainingDataframeList.append(df[mask_selected])

test_df = None # Clear memory
trainingDataframeList = None # Clear memory

# Lambda function to round all of the wavenumbers so that column labels are all matching, and then concatenate all the datasets together
rawTrainingDataframe = pd.concat(
    (
        df.rename(
            columns=lambda c: round(float(c), 1) if c not in df.columns[:last_nonwavenum_idx] else c
        )
        for df in masked_trainingDataframeList
    ),
    ignore_index=True
)

masked_trainingDataframeList = None # Clear memory

interpRawTrainingDataframe = pd.DataFrame()
interpDataFramelist=[]
# Interpolate the spectra to increase the accuracy of the model
for index, row in rawTrainingDataframe.iterrows():
    print(index)
    row = row[last_nonwavenum_idx:]
    frequencies = row.index.to_numpy()
    frequencies = frequencies[::-1]  # Reverse the order for interpolation
    spectrum = row.to_numpy()
    spectrum = spectrum[::-1]  # Reverse the order for interpolation
    interpolated_wavenumber, interpolated_spectrum = interpolate_spectrum(frequencies, spectrum, low=900, high=1800)
    #interpRawTrainingDataframe = pd.concat([interpRawTrainingDataframe, pd.DataFrame(interpolated_spectrum, columns=interpolated_wavenumber)], ignore_index=True)
    interpDataFramelist.append(pd.DataFrame([interpolated_spectrum], columns=interpolated_wavenumber))

interpRawTrainingDataframe = pd.concat(interpDataFramelist, ignore_index=True)
interpDataFramelist = None # Clear memory

print(interpRawTrainingDataframe)

"""
K-fold Validation

*** Need to rework this section to properly implement k-fold validation for the interpolated data ***

unique_samples = rawTrainingDataframe["Sample Name"].unique()
n_folds = 5

fold_map = {}
for i, sid in enumerate(unique_samples):
    fold_map[sid] = i % n_folds

rawTrainingDataframe["fold"] = rawTrainingDataframe["Sample Name"].map(fold_map)
"""


"""
Initialize the model, train the model and save. 
"""
wavenumbers = sorted(wavenumbers)
betaVAE_trainingData = rawTrainingDataframe[wavenumbers]
input_dim = len(wavenumbers)
output_dim = input_dim

latent_dim = 16
beta = 10.0

epochs = 3

encoder = build_encoder(input_dim, latent_dim)
decoder = build_decoder(latent_dim, output_dim)

vae = BetaVAE(encoder, decoder, beta=beta)

vae.compile(optimizer=keras.optimizers.Adam(learning_rate=1e-4))

vae.build(input_shape=(None,505))

print(vae.summary())

print(vae.get_config())

#for array in betaVAE_trainingData.values:
    #print([wavenumbers, array])
array = np.asarray(betaVAE_trainingData.values, dtype='float32')
#array = array[None,:]
vae.fit(array, epochs=epochs, batch_size=128)

tf.saved_model.save(vae, "./new_vae/")
tf.saved_model.save(encoder, './new_encoder/')
tf.saved_model.save(decoder, './new_decoder/')

saved_model = tf.saved_model.load("./new_vae/")

print(saved_model.signatures)

if saved_model is not None:
    print("Model loaded successfully!")

else:
    print("Failed to load the model.")
