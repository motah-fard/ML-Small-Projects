import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

#import data
reviews = pd.read_csv("reviews.csv")

#print column names
columns = reviews.columns
print(reviews.columns)
 
#print .info
print(reviews.info())

#look at the counts of recommended

print(reviews['recommended'].value_counts())

#create binary dictionary
binary_dict = {True: 1, False: 0}
 # ta inja oumadam
#transform column
reviews['recommended'] = reviews['recommended'].map(binary_dict)

 
#print your transformed column
print(reviews['recommended'])

#look at the counts of rating
print(reviews['rating'].value_counts())
 
#create dictionary
rating_dict = {'Hated it':1,  'Not great':2, 'Was okay':3, 'Liked it':4, 'Loved it':5}

 
#transform rating column
reviews['rating'] = reviews['rating'].map(rating_dict)

 
#print your transformed column values

print(reviews['rating'].value_counts())
#get the number of categories in a feature

print(reviews['department_name'].value_counts())

#perform get_dummies
one_hot = pd.get_dummies(reviews['department_name'])

 
#join the new columns back onto the original

reviews = reviews.join(one_hot)
#print column names

print(reviews.columns)
#transform review_date to date-time data

reviews['reviewed_date'] = pd.to_datetime(reviews['review_date'])

#print review_date data type 
print("review_date data type:", reviews['reviewed_date'].dtype)

#get numerical columns
reviews = reviews[['clothing_id', 'age', 'recommended', 'rating', 'Bottoms', 'Dresses', 'Intimate', 'Jackets', 'Tops', 'Trend']].copy()
#reset index
reviews = reviews.set_index('clothing_id')
scaler = StandardScaler()
scaler.fit_transform(reviews)

