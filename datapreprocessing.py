import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer





df1 = pd.read_csv('file.csv')
df1.drop(columns=['date'], inplace=True)
df1.drop(columns=['pc'], inplace=True)
df1.drop(columns=['id'], inplace=True)
df1.drop(columns=['content'], inplace=True)
df1.to_csv('file1.csv', index=False)

df2 = pd.read_csv('device.csv')
df2.drop(columns=['pc'], inplace=True)
df2.drop(columns=['id'], inplace=True)
df2.to_csv('device1.csv', index=False)





# Define chunk size
chunk_size = 100

# Initialize an empty DataFrame to store the merged result
merged_df = pd.DataFrame()

# Load and merge files in chunks
for chunk1, chunk2, chunk3 in zip(pd.read_csv('logon.csv', chunksize=chunk_size),pd.read_csv('file1.csv', chunksize=chunk_size),pd.read_csv('device1.csv', chunksize=chunk_size)):
    # Merge chunks based on a common column
    merged_chunk = pd.merge(chunk1, chunk2, on='user')
    merged_chunk = pd.merge(merged_chunk, chunk3, on='user')
    
    

    
    # Append merged chunk to the result
    merged_df = pd.concat([merged_df, merged_chunk])

# Save merged DataFrame to a new CSV file
merged_df.to_csv('final.csv', index=False)


dfd = pd.read_csv('final.csv')


# Convert the time column to datetime format
dfd['datel'] = pd.to_datetime(dfd['datel'])
dfd['dated'] = pd.to_datetime(dfd['dated'])


# Function to label rows based on logon and connect times
def label_data(row):
    logon_hour = row['datel'].hour
    connect_hour = row['dated'].hour
    if (logon_hour < 9 or logon_hour > 18) and (connect_hour < 9 or connect_hour > 18):
        return 1
    else:
        return 0



# Apply the function to create the 'Label' column
dfd['label'] = dfd.apply(label_data, axis=1)

# If you want to save the result to a new CSV
dfd.to_csv("final.csv", index=False)
def label_exe(row):
    if ".exe" in str(row["filename"]):
        return 1
    else:
        return row["label"]

dfd["label"] = dfd.apply(label_exe, axis=1)
dfd.to_csv("final.csv", index=False)

