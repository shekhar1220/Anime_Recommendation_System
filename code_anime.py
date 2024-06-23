#!/usr/bin/env python
# coding: utf-8

# In[7]:


get_ipython().system('pip install pandas')
get_ipython().system('pip install scikit-surprise')
import surprise


# In[24]:


import pandas as pd
from surprise import Dataset, Reader, SVD
from surprise.model_selection import train_test_split, cross_validate

# Load the merged dataset
data = pd.read_csv("C:\\Users\\ASUS\\OneDrive\\Desktop\\My_Project\\archive\\Anime.csv")

# Display the first few rows of the dataset
#print(data.head())
print(data.columns)

# Extract unique anime titles
anime_titles = data[['anime_id', 'name']].drop_duplicates()

# Prepare data for the Surprise library
reader = Reader(rating_scale=(1, 10))
dataset = Dataset.load_from_df(data[['user_id', 'anime_id', 'rating']], reader)

# Train-test split
trainset, testset = train_test_split(dataset, test_size=0.2, random_state=42)

# Model training
model = SVD()
model.fit(trainset)

# Model evaluation
cross_validate(model, dataset, measures=['RMSE', 'MAE'], cv=5, verbose=True)

# Function to get top N recommendations
def get_top_n_recommendations(user_id, model, anime_df, n=10):
    # Get a list of all anime IDs
    anime_ids = anime_df['anime_id'].unique()
    
    # Predict ratings for the given user and all anime
    user_ratings = [(anime_id, model.predict(user_id, anime_id).est) for anime_id in anime_ids]
    
    # Sort the predictions by rating in descending order
    user_ratings.sort(key=lambda x: x[1], reverse=True)
    
    # Get the top N recommendations
    top_n_recommendations = user_ratings[:n]
    
    # Get the titles of the top N recommendations
    recommendations = [(anime_df[anime_df['anime_id'] == anime_id]['name'].values[0], rating) 
                       for anime_id, rating in top_n_recommendations]
    
    return recommendations


# Get top 10 recommendations for a user
user_id = 1
recommendations = get_top_n_recommendations(user_id, model, anime_titles)
print(f'Top 10 recommendations for user {user_id}:')
for i, (anime_title, rating) in enumerate(recommendations, 1):
    print(f'{i}. {anime_title} (Predicted Rating: {rating:.2f})')


# In[ ]:





# In[ ]:




