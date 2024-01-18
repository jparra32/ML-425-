#!/usr/bin/env python
# coding: utf-8

# In[5]:


import numpy as np

from numpy import array
from keras.models import Sequential
from keras.layers import LSTM
from keras.layers import Dense


# In[6]:


from sklearn.datasets import fetch_california_housing


# In[6]:


ca_house_db=fetch_california_housing()
print (ca_house_db.data.shape)
print (ca_house_db.target.shape)
print (ca_house_db.feature_names)
print (ca_house_db.DESCR)
print (ca_house_db)


# In[10]:


housing = fetch_california_housing()
X = housing.data
y = housing.target


# In[15]:


from sklearn.model_selection import train_test_split


# In[16]:


# Split the data into training (50%) and validation (50%) sets
X_train_val, X_test, y_train_val, y_test = train_test_split(housing.data, housing.target, test_size=0.5, random_state=42)

# Split the training set into training (80%) and validation (20%) subsets
X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val, test_size=0.2, random_state=42)

# Print the shapes of the resulting subsets
print("Training set: X_train = {}, y_train = {}".format(X_train.shape, y_train.shape))
print("Validation set: X_val = {}, y_val = {}".format(X_val.shape, y_val.shape))
print("Testing set: X_test = {}, y_test = {}".format(X_test.shape, y_test.shape))


# In[9]:


import tensorflow as tf
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

# Load data
housing = fetch_california_housing()
X_train_full, X_test, y_train_full, y_test = train_test_split(housing.data, housing.target, random_state=42)
X_train, X_valid, y_train, y_valid = train_test_split(X_train_full, y_train_full, random_state=42)

# Define model
model = tf.keras.models.Sequential([
    tf.keras.layers.Dense(64, activation="relu", input_shape=X_train.shape[1:]),
    tf.keras.layers.Dense(64, activation="relu"),
    tf.keras.layers.Dense(64, activation="relu"),
    tf.keras.layers.Dense(1, activation="linear")
])

# Compile model
model.compile(loss="mse", optimizer="adam")

# Train model
history = model.fit(X_train, y_train, epochs=100, validation_data=(X_valid, y_valid))

# Predict on validation set
y_valid_pred = model.predict(X_valid)

# Calculate R2 score on validation set
r2 = r2_score(y_valid, y_valid_pred)
print(f"Set 1 R2 score: {r2}")


# In[1]:


import tensorflow as tf
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

# Load data
housing = fetch_california_housing()
X_train_full, X_test, y_train_full, y_test = train_test_split(housing.data, housing.target, random_state=42)
X_train, X_valid, y_train, y_valid = train_test_split(X_train_full, y_train_full, random_state=42)

# Define model
model = tf.keras.models.Sequential([
    tf.keras.layers.Dense(128, activation="sigmoid", input_shape=X_train.shape[1:]),
    tf.keras.layers.Dense(128, activation="sigmoid"),
    tf.keras.layers.Dense(1, activation="linear")
])

# Compile model
model.compile(loss="mse", optimizer="sgd")

# Train model
history = model.fit(X_train, y_train, epochs=100, validation_data=(X_valid, y_valid))

# Predict on validation set
y_valid_pred = model.predict(X_valid)

# Calculate R2 score on validation set
r2 = r2_score(y_valid, y_valid_pred)
print(f"Set 2 R2 score: {r2}")


# In[10]:


from sklearn.metrics import r2_score

# Define the model
model = tf.keras.models.Sequential([
    tf.keras.layers.Dense(64, activation='relu', input_shape=(8,)),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(1)
])

# Compile the model
model.compile(optimizer='adam', loss='mse', metrics=['mae'])

# Train the model on the entire training set
model.fit(X_train, y_train, epochs=50, batch_size=128, verbose=0)

# Evaluate the model on the testing set
y_pred = model.predict(X_test)
r2 = r2_score(y_test, y_pred)
print("R2 score on testing set:", r2)


# In[11]:


from sklearn.metrics import r2_score

# Define the model
model = tf.keras.models.Sequential([
    tf.keras.layers.Dense(64, activation='relu', input_shape=(8,)),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(1)
])

# Compile the model
model.compile(optimizer='adam', loss='mse', metrics=['mae'])

# Train the model on the entire training set
model.fit(X_train, y_train, epochs=50, batch_size=128, verbose=0)

# Evaluate the model on the testing set
y_pred = model.predict(X_test)
r2 = r2_score(y_test, y_pred)
print("R2 score on testing set:", r2)


# In[12]:


# Apply top-ranked model over testing samples
y_pred_test = model.predict(X_test)

# Calculate absolute errors
abs_errors = np.abs(y_test - y_pred_test).flatten()

# Sort errors in descending order and select top 10
largest_errors_idx = np.argsort(abs_errors)[::-1][:10]
largest_errors = abs_errors[largest_errors_idx]


# The R2 score on the testing set for the selected model is 0.525, which indicates a good fit of the model on the testing data.

# In[ ]:




