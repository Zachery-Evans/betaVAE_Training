import os
import re
import numpy as np
import pandas as pd
from spectrum_preprocessing import roundWavenumbers, distribution_Selection, pipeline
"""
Load and Format the TESTING data 

"""
stdDevs = 3
print("Loading and preprocessing testing data...")
if not os.path.exists('./interpolated_testing_data.csv'):
    testing_df = pd.read_csv('./spectral_data/SMP65#013 35d 920um.csv', low_memory=False, skiprows=[1,2])
    testing_df = roundWavenumbers(testing_df)

    last_nonwavenum_idx = testing_df.columns.get_loc('1981.7 - 2095.8') + 1
    wavenumbers = testing_df.columns[last_nonwavenum_idx:]
    
    _, _, mask_selected, modePosition, areaPE = distribution_Selection(testing_df, '1981.7 - 2095.8', stdDevs)
    testing_df = testing_df[mask_selected]

    interpRawTestingDataframe = pd.DataFrame()
    interpDataFramelist = []
    for index, row in testing_df.iterrows():
        row = row[last_nonwavenum_idx:]

        frequencies = row.index.to_numpy(dtype=float)
        frequencies = frequencies[::-1]  # Reverse the order for interpolation
        spectrum = row.to_numpy(dtype=float)
        spectrum = spectrum[::-1]  # Reverse the order for interpolation
        
        frequencies, spectrum = pipeline(frequencies, spectrum)

        interpDataFramelist.append(pd.DataFrame(data=[spectrum], columns=frequencies))

    testing_df = pd.concat(interpDataFramelist, ignore_index=True)
    testing_df.fillna(testing_df.mean(), inplace=True)
    interpDataFramelist = None # Clear memory

    testing_df.to_csv("interpolated_testing_data.csv", index=False)
    print("Testing data preprocessing and interpolation complete, CSV file created.")
else:
    print("CSV File already exists.")
    testing_df = pd.read_csv("interpolated_testing_data.csv", low_memory=False)


"""
Load and Format the TRAINING data

"""
print("Loading and preprocessing training data...")
if not os.path.exists('./interpolated_training_data.csv'):
    path = './training_data/'
    #List all of the files in the data directory.
    allFiles = os.listdir(path)
    # Take only the files that contain data pertaining to SMP65#010 
    trainingFiles = [file for file in allFiles if file.endswith('.csv') and 'SMP65#010' in file and 'full width' not in file]

    trainingFiles = sorted(trainingFiles, key=lambda x: int(re.search(r'(?<= )(.+?)(?=d)', x).group()))

    # Read all of the data and place the dataframes into a list
    trainingDataframeList = [pd.read_csv(path+file, low_memory=False, skiprows=[1,2]) for file in trainingFiles]

    print(trainingFiles)

    last_nonwavenum_idx = trainingDataframeList[0].columns.get_loc('1981.7 - 2095.8') + 1
    wavenumbers = trainingDataframeList[0].columns[last_nonwavenum_idx:]
    
    masked_trainingDataframeList = []

    for df in trainingDataframeList:
        _, _, mask_selected, modePosition, areaPE = distribution_Selection(df, '1981.7 - 2095.8', stdDevs)
        masked_trainingDataframeList.append(df[mask_selected])
    trainingDataframeList = None # Clear memory

    # Lambda function to round all of the wavenumbers so that column labels are all matching, 
    # and then concatenate all the datasets together
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

    training_df = pd.DataFrame()
    interpDataFramelist=[]

    # Interpolate the spectra to increase the accuracy of the model
    for index, row in rawTrainingDataframe.iterrows():

        row = row[last_nonwavenum_idx:]

        frequencies = row.index.to_numpy(dtype=float)
        frequencies = frequencies[::-1]  # Reverse the order for interpolation
        spectrum = row.to_numpy(dtype=float)
        spectrum = spectrum[::-1]  # Reverse the order for interpolation

        frequencies, spectrum = pipeline(frequencies, spectrum)

        interpDataFramelist.append(pd.DataFrame(data=[spectrum], columns=frequencies))

    training_df = pd.concat(interpDataFramelist, ignore_index=True)
    training_df.fillna(training_df.mean(), inplace=True)
    
    interpDataFramelist = None # Clear memory
    training_df.to_csv("interpolated_training_data.csv", index=False)
    frequencies = training_df.columns.astype(float)

    print("Training data preprocessing and interpolation complete, CSV file created.")

else:
    print("CSV File already exists.")
    training_df = pd.read_csv("interpolated_training_data.csv", low_memory=False)
    frequencies = training_df.columns.astype(float)

