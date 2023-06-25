#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().system('pip freeze | grep scikit-learn')


# In[25]:


import pickle
import pandas as pd
import math
import os



# In[3]:


with open('model.bin', 'rb') as f_in:
    dv, model = pickle.load(f_in)


# In[4]:


categorical = ['PULocationID', 'DOLocationID']

def read_data(filename):
    df = pd.read_parquet(filename)
    
    df['duration'] = df.tpep_dropoff_datetime - df.tpep_pickup_datetime
    df['duration'] = df.duration.dt.total_seconds() / 60

    df = df[(df.duration >= 1) & (df.duration <= 60)].copy()

    df[categorical] = df[categorical].fillna(-1).astype('int').astype('str')
    
    return df


# In[5]:


df = read_data('https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_2022-02.parquet')


# In[6]:


dicts = df[categorical].to_dict(orient='records')
X_val = dv.transform(dicts)
y_pred = model.predict(X_val)


# In[9]:


print(y_pred)


# In[11]:


def calc_std_dev(y_pred):
    n = len(y_pred)
    mean = sum(y_pred) / n
    squared_diff_sum = sum((x - mean) ** 2 for x in y_pred)
    variance = squared_diff_sum / n
    std_dev = math.sqrt(variance)
    return std_dev


# In[14]:


std_dev = calc_std_dev(y_pred)


# In[15]:


print("Standard Deviation:", std_dev)


# In[19]:


year = 2022
month = 2


# In[21]:


# Create an artificial ride_id column
df['ride_id'] = f'{year:04d}/{month:02d}_' + df.index.astype('str')

# Prepare the dataframe with results
df_result = pd.DataFrame({'ride_id': df['ride_id'], 'predicted_duration': y_pred})

# Save the dataframe as a Parquet file
output_file = 'results.parquet'
df_result.to_parquet(output_file, engine='pyarrow', compression=None, index=False)


# In[26]:


import os

output_file = 'results.parquet'

file_size_bytes = os.path.getsize(output_file)
file_size_mb = file_size_bytes / (1024 * 1024)

print("Size of the output file:", file_size_mb, "MB")


# In[ ]:




