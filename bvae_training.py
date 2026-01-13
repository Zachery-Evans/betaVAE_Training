import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["KERAS BACKEND"] = "tensorflow"
import re
import pandas as pd
import numpy as np
import tensorflow as tf
import keras
from keras import layers
from scipy.stats import gaussian_kde
from sklearn.model_selection import cross_val_score, KFold
import spectrum_preprocessing as sp
from bvae_model import pipeline

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
Reparameterization 

"""
class Sampling(keras.layers.Layer):
    """Uses (z_mean, z_log_var) to sample z, the vector encoding a digit."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.seed_generator = tf.random.set_seed(1337)

    def call(self, inputs):

        z_mean, z_log_var = inputs
        batch = tf.shape(z_mean)[0]
        dim = tf.shape(z_mean)[1]

        epsilon = keras.backend.random_normal(
            shape=(batch, dim),
            seed=self.seed_generator
        )

        """
        z_mean, z_log_var = args
        set = tf.shape(z_mean)[0]
        batch = tf.shape(z_mean)[1]
        dim = tf.shape(z_mean)[-1]
        # by default, random_normal has mean=0 and std=1.0
        epsilon = tf.random.normal(shape=(set, dim))#tfp.distributions.Normal(mean=tf.zeros(shape=(batch, dim)),loc=tf.ones(shape=(batch, dim)))
        return z_mean + (z_log_var * epsilon)
        """

        return z_mean + keras.backend.exp(0.5 * z_log_var) * epsilon

    

"""
BetaVAE Model

"""
class BetaVAE(keras.Model):
    def __init__(self, encoder, decoder, beta, **kwargs):
        super().__init__(**kwargs)
        self.encoder = encoder # ENCODER HAS THE SAMPLING LAYER INSIDE
        self.decoder = decoder
        self.beta = beta
        self.seed_generator = tf.random.set_seed(1337)

        self.total_loss_tracker = keras.metrics.Mean(name="loss")
        self.recon_loss_tracker = keras.metrics.Mean(name="recon_loss")
        self.kl_loss_tracker = keras.metrics.Mean(name="kl_loss")
    
    def call(self, data):
        # Forward pass
        with tf.GradientTape() as tape:
            x = self.encoder(data)
            z_mean, z_logvar, z = tf.split(x, num_or_size_splits=3)
            reconstruction = self.decoder(z)

            recon_loss = tf.reduce_mean(
                tf.reduce_sum(keras.losses.binary_crossentropy(data, reconstruction))
            )

            kl_loss = -0.5 * tf.reduce_mean(
                tf.reduce_sum(1 + z_logvar - tf.square(z_mean) - tf.exp(z_logvar))
            )

            total_loss = recon_loss + self.beta * kl_loss

            grads = tape.gradient(total_loss, self.trainable_weights)
            self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
            self.recon_loss_tracker.update_state(recon_loss)
            self.kl_loss_tracker.update_state(kl_loss)

        return recon_loss, kl_loss, total_loss

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

    """
    The training of the Model, here we have the minimization of the loss function and define the loss
    function of the beta VAE to include the beta normalization term.
    """
    @tf.function
    def train_step(self, data):
        
        if isinstance(data, tuple):
            data = data[0]

        recon_loss, kl_loss, total_loss = self.call(data)

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
trainingFiles = [file for file in allFiles if file.endswith('.csv') and 'SMP65#010' in file and 'full width' not in file]

trainingFiles = sorted(trainingFiles, key=lambda x: int(re.search(r'(?<= )(.+?)(?=d)', x).group()))

# Read all of the data and place the dataframes into a list
trainingDataframeList = [pd.read_csv(path+file, low_memory=False, skiprows=[1,2]) for file in trainingFiles]

testingModelFlag = True
if testingModelFlag:
    for i, dataframe in enumerate(trainingDataframeList):
        trainingDataframeList[i] = dataframe.sample(n=1000, random_state=42).reset_index(drop=True)  

test_df = trainingDataframeList[0]
test_df = roundWavenumbers(test_df)

last_nonwavenum_idx = test_df.columns.get_loc('1981.7 - 2095.8') + 1
wavenumbers = test_df.columns[last_nonwavenum_idx:]

masked_trainingDataframeList = []

stdDevs = 2

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
print("Preprocessing and interpolating the training data...")
# Interpolate the spectra to increase the accuracy of the model
for index, row in rawTrainingDataframe.iterrows():

    row = row[last_nonwavenum_idx:]

    frequencies = row.index.to_numpy()
    frequencies = frequencies[::-1]  # Reverse the order for interpolation
    spectrum = row.to_numpy()
    spectrum = spectrum[::-1]  # Reverse the order for interpolation
    
    frequencies, spectrum = pipeline(frequencies, spectrum)

    interpDataFramelist.append(pd.DataFrame(data=[spectrum], columns=frequencies))

interpRawTrainingDataframe = pd.concat(interpDataFramelist, ignore_index=True)
interpDataFramelist = None # Clear memory

#print(interpRawTrainingDataframe)

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
wavenumbers = sorted(frequencies)
betaVAE_trainingData = interpRawTrainingDataframe[wavenumbers]
input_dim = len(wavenumbers)
output_dim = input_dim

batch=128

latent_dim = 16
beta = 10.0

epochs = 15

array = np.asarray(betaVAE_trainingData.values, dtype=np.float32)

"""
Build the Encoder
"""
input = keras.Input(shape=input_dim, name='spectra_input') 
x = layers.Dense(batch, activation='relu')(input) 
x = layers.Dense(batch, activation='relu')(x) 
z_mean = layers.Dense(latent_dim, name='z_mean')(x) 
z_logvar = layers.Dense(latent_dim, name='z_logvar')(x) 
z = Sampling()([z_mean, z_logvar])
encoder = keras.Model(inputs=input, outputs=[z_mean, z_logvar, z], name='encoder')
encoder.summary()

"""
Build the Decoder 
"""
latent_inputs = keras.Input(shape=(latent_dim,), name='latent_variables')
x = layers.Dense(batch, activation='relu')(latent_inputs)
outputs = layers.Dense(output_dim, activation='linear')(x)
decoder = keras.Model([latent_inputs], outputs, name='decoder')
decoder.summary()

vae = BetaVAE(encoder, decoder, beta=beta)

vae.compile(optimizer=keras.optimizers.Adam(learning_rate=1e-4))

test_input = tf.random.normal(shape=(latent_dim, input_dim))  # Create a test input the shape (latent_dim, input_dim)

vae(test_input)  # Build the model by calling it on a test input
vae.fit(x=array, epochs=epochs, batch_size=batch)

tf.saved_model.save(vae, "./new_vae/")
tf.saved_model.save(encoder, './new_encoder/')
tf.saved_model.save(decoder, './new_decoder/')

saved_encoder = tf.saved_model.load("./new_encoder/")
saved_decoder = tf.saved_model.load("./new_decoder/")
saved_vae = tf.saved_model.load("./new_vae/")

#print(saved_vae.signatures)

if saved_vae is not None and saved_encoder is not None and saved_decoder is not None:
    print("All models loaded successfully!")

else:
    print("Failed to load the model.")
