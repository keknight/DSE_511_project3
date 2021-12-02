"""
Title: Data preprocessing
Author: Anna-Maria nau

This file contains source code to perform data preprocessing.
"""

import os
import pickle as pkl

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA



def data_preprocessing():
	'''
	This function performs basic data preprocessing, such as changing column names, dropping features,
	splitting data in train and test sets, min-max normalization, and principal component analysis (PCA)
	for feature reduction. The processed data will be saved into ../data/processed/.
	'''

	# load data
	print('Loading raw data...')
	with open(os.path.join('..', 'data', 'raw', 'nasa.csv'), 'rb') as input_file:
		data = pd.read_csv(input_file)
	print('Shape of raw data:', data.shape)

	# fix column names: replace each ' ' with '_'.
	data.columns = data.columns.str.strip().str.lower().str.replace(' ', '_', regex=False)\
		.str.replace('(', '_', regex=False).str.replace(')', '', regex=False)

	# drop redundant and unnecessary features
	data.drop(['equinox', 'orbiting_body', 'name', 'neo_reference_id', 'est_dia_in_km_max', 'est_dia_in_m_min',
			   'est_dia_in_m_max', 'est_dia_in_m_min', 'est_dia_in_m_max', 'est_dia_in_miles_min',
			   'est_dia_in_miles_max', 'est_dia_in_feet_min', 'est_dia_in_feet_max', 'relative_velocity_km_per_sec',
			   'miles_per_hour', 'close_approach_date', 'epoch_date_close_approach', 'orbit_determination_date',
			   'miss_dist._astronomical', 'miss_dist._lunar', 'miss_dist._miles'], axis=1, inplace=True)
	print('Shape of data after removing features:', data.shape)

	# split data randomly in train and test sets (80/20 split)
	X_train, X_test, y_train, y_test = train_test_split(data.iloc[:, :-1], data.hazardous, test_size=0.2, random_state=0)
	print('Splitting data into train/test set (80:20 ratio)...')
	print(f'   X_train shape: {X_train.shape}')
	print(f'   y_train shape: {y_train.shape}')
	print(f'   X_test shape: {X_test.shape}')
	print(f'   y_test shape: {y_test.shape}')

	print('Applying min-max scaling...')
	# define standard scaler
	scaler = MinMaxScaler()
	# get scale parameters from train set
	scaler.fit(X_train)
	# scale data
	X_train_scaled = scaler.transform(X_train)
	X_train_scaled = pd.DataFrame(X_train_scaled, columns=['absolute_magnitude', 'est_dia_in_km_min',
														   'relative_velocity_km_per_hr', 'miss_dist._kilometers',
														   'orbit_id',
														   'orbit_uncertainity', 'minimum_orbit_intersection',
														   'jupiter_tisserand_invariant', 'epoch_osculation',
														   'eccentricity',
														   'semi_major_axis', 'inclination', 'asc_node_longitude',
														   'orbital_period', 'perihelion_distance', 'perihelion_arg',
														   'aphelion_dist', 'perihelion_time', 'mean_anomaly',
														   'mean_motion'])
	X_test_scaled = scaler.transform(X_test)
	X_test_scaled = pd.DataFrame(X_test_scaled, columns=['absolute_magnitude', 'est_dia_in_km_min',
														 'relative_velocity_km_per_hr', 'miss_dist._kilometers',
														 'orbit_id',
														 'orbit_uncertainity', 'minimum_orbit_intersection',
														 'jupiter_tisserand_invariant', 'epoch_osculation',
														 'eccentricity',
														 'semi_major_axis', 'inclination', 'asc_node_longitude',
														 'orbital_period', 'perihelion_distance', 'perihelion_arg',
														 'aphelion_dist', 'perihelion_time', 'mean_anomaly',
														 'mean_motion'])

	print('Applying PCA with 95% explained variance to scaled data...')
	pca = PCA(n_components=0.95, svd_solver='full', random_state=0)
	X_train_scaled_pca = pca.fit_transform(X_train_scaled)
	X_train_scaled_pca = pd.DataFrame(X_train_scaled_pca)
	X_test_scaled_pca = pca.transform(X_test_scaled)
	X_test_scaled_pca = pd.DataFrame(X_test_scaled_pca)
	print(f'   Input feature space was reduced from {X_train.shape[1]} to {X_train_scaled_pca.shape[1]} principal components')

	print('Saving processed data for modeling...')
	# Save normalized data
	with open("../data/processed/train_scaled.pkl", "wb") as f:
		pkl.dump([X_train_scaled, y_train], f)

	with open("../data/processed/test_scaled.pkl", "wb") as f:
		pkl.dump([X_test_scaled, y_test], f)

	# Save normalized + PCA data
	with open("../data/processed/train_scaled_pca.pkl", "wb") as f:
		pkl.dump([X_train_scaled_pca, y_train], f)

	with open("../data/processed/test_scaled_pca.pkl", "wb") as f:
		pkl.dump([X_test_scaled_pca, y_test], f)
