#!/usr/bin/env python
# coding: utf-8

# # Movie Recommendations HW

# **Name:**  

# **Collaboration Policy:** Homeworks will be done individually: each student must hand in their own answers. Use of partial or entire solutions obtained from others or online is strictly prohibited.

# **Late Policy:** Late submission have a penalty of 2\% for each passing hour. 

# **Submission format:** Successfully complete the Movie Lens recommender as described in this jupyter notebook. Submit a `.py` and an `.ipynb` file for this notebook. You can go to `File -> Download as ->` to download a .py version of the notebook. 
# 
# **Only submit one `.ipynb` file and one `.py` file.** The `.ipynb` file should have answers to all the questions. Do *not* zip any files for submission. 

# **Download the dataset from here:** https://grouplens.org/datasets/movielens/1m/

# In[2]:


# Import all the required libraries
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings("ignore")


# ## Reading the Data
# Now that we have downloaded the files from the link above and placed them in the same directory as this Jupyter Notebook, we can load each of the tables of data as a CSV into Pandas. Execute the following, provided code.

# In[3]:


# Read the dataset from the two files into ratings_data and movies_data
column_list_ratings = ["UserID", "MovieID", "Ratings","Timestamp"]
ratings_data  = pd.read_csv('ratings.dat',sep='::',names = column_list_ratings ,engine='python', encoding='latin-1')
column_list_movies = ["MovieID","Title","Genres"]
movies_data = pd.read_csv('movies.dat',sep = '::',names = column_list_movies ,engine = 'python', encoding='latin-1')
column_list_users = ["UserID","Gender","Age","Occupation","Zixp-code"]
user_data = pd.read_csv("users.dat",sep = "::",names = column_list_users ,engine = 'python', encoding='latin-1')


# `ratings_data`, `movies_data`, `user_data` corresponds to the data loaded from `ratings.dat`, `movies.dat`, and `users.dat` in Pandas.

# ## Data analysis

# We now have all our data in Pandas - however, it's as three separate datasets! To make some more sense out of the data we have, we can use the Pandas `merge` function to combine our component data-frames. Run the following code:

# In[4]:


data=pd.merge(pd.merge(ratings_data,user_data),movies_data)
data


# Next, we can create a pivot table to match the ratings with a given movie title. Using `data.pivot_table`, we can aggregate (using the average/`mean` function) the reviews and find the average rating for each movie. We can save this pivot table into the `mean_ratings` variable. 

# In[5]:


mean_ratings=data.pivot_table('Ratings','Title',aggfunc='mean')
mean_ratings


# Now, we can take the `mean_ratings` and sort it by the value of the rating itself. Using this and the `head` function, we can display the top 15 movies by average rating.

# In[6]:


mean_ratings=data.pivot_table('Ratings',index=["Title"],aggfunc='mean')
top_15_mean_ratings = mean_ratings.sort_values(by = 'Ratings',ascending = False).head(15)
top_15_mean_ratings


# Let's adjust our original `mean_ratings` function to account for the differences in gender between reviews. This will be similar to the same code as before, except now we will provide an additional `columns` parameter which will separate the average ratings for men and women, respectively.

# In[7]:


mean_ratings=data.pivot_table('Ratings',index=["Title"],columns=["Gender"],aggfunc='mean')
mean_ratings


# We can now sort the ratings as before, but instead of by `Rating`, but by the `F` and `M` gendered rating columns. Print the top rated movies by male and female reviews, respectively.

# In[8]:


data=pd.merge(pd.merge(ratings_data,user_data),movies_data)

mean_ratings=data.pivot_table('Ratings',index=["Title"],columns=["Gender"],aggfunc='mean')
top_female_ratings = mean_ratings.sort_values(by='F', ascending=False)
print(top_female_ratings.head(15))

top_male_ratings = mean_ratings.sort_values(by='M', ascending=False)
print(top_male_ratings.head(15))


# In[9]:


mean_ratings['diff'] = mean_ratings['M'] - mean_ratings['F']
sorted_by_diff = mean_ratings.sort_values(by='diff')
sorted_by_diff[:10]


# Let's try grouping the data-frame, instead, to see how different titles compare in terms of the number of ratings. Group by `Title` and then take the top 10 items by number of reviews. We can see here the most popularly-reviewed titles.

# In[10]:


ratings_by_title=data.groupby('Title').size()
ratings_by_title.sort_values(ascending=False).head(10)


# Similarly, we can filter our grouped data-frame to get all titles with a certain number of reviews. Filter the dataset to get all movie titles such that the number of reviews is >= 2500.

# ## Question 1

# Create a ratings matrix using Numpy. This matrix allows us to see the ratings for a given movie and user ID. Every element $[i,j]$ is a rating by user $i$ for movie $j$. Print the **shape** of the matrix produced.
# 
# **Notes:**
# - Do *not* use `pivot_table`.
# - A ratings matrix is *not* the same as `ratings_data` from above.
# - If you're stuck, you might want to look into the `np.ndarray` datatype and how to create one of the desired shape.
# - Every review lies between 1 and 5, and thus fits within a `uint8` datatype, which you can specify to numpy.

# In[38]:


# Create the matrix
ratings_matrix = np.ndarray(shape=(data.UserID.max(),
                                   data.MovieID.max()), 
                                   dtype=np.uint8)


# In[39]:


ratings_matrix[ratings_data.UserID.values - 1, ratings_data.MovieID.values - 1] = ratings_data.Ratings.values
ratings_matrix.shape


# ## Question 2

# Normalize the ratings matrix (created in **Question 1**) using Z-score normalization. While we can't use `sklearn`'s `StandardScaler` for this step, we can do the statistical calculations ourselves to normalize the data.
# 
# Before you start:
# - All of the `NaN` values in the dataset should be replaced with the average rating for the given movie. This is a complex topic, but for our case replacing empty values with the mean will make it so that the absence of a rating doesn't affect the overall average, and it provides an "expected value" which is useful for computing correlations and recommendations in later steps. 
# - Your first step should be to get the average of every *column* of the ratings matrix (we want an average by title, not just by user!).
# - Second, we want to subtract the average from the original ratings thus allowing us to get a mean of 0 in every row. It may be very close but not exactly zero because of the limited precision `float`s allow.
# - Lastly, divide this by the standard deviation of the *column*
# 
# - While creating the ratings matrix, you might not get any NaN values but if you look closely, you will see 0 values. This should be treated as NaN values as the original data does not have 0 rating. This should be replaced by the average.

# In[40]:


colums_mean = np.nanmean(ratings_matrix, axis=0)
nan_indices = np.where(np.isnan(ratings_matrix))
ratings_matrix[nan_indices] = np.take(colums_mean, nan_indices[1])
ratings_matrix = (ratings_matrix - ratings_matrix.mean(axis = 0))/ratings_matrix.std(axis = 0) #normalize the data
ratings_matrix = np.where(np.isnan(ratings_matrix), 0, ratings_matrix)
ratings_matrix


# ## Question 3

# We're now going to perform Singular Value Decomposition (SVD) on the normalized ratings matrix from the previous question. Perform the process using numpy, and along the way print the shapes of the $U$, $S$, and $V$ matrices you calculated.

# In[41]:


# Compute the SVD of the normalised matrix
U, S, V= np.linalg.svd(ratings_matrix, full_matrices=True)


# In[42]:


# Print the shapes
U.shape, S.shape, V.shape


# ## Question 4

# Reconstruct four rank-k rating matrix $R_k$, where $R_k = U_kS_kV_k^T$ for k = [100, 1000, 2000, 3000]. Using each of $R_k$ make predictions for 3 users (select them from the dataset) for the movie with ID 1377 (Batman Returns).

# In[43]:


def reconstruct_rank_k_matrix(U, S, V, k):
    sigma_k = np.diag(S)[:k,:k] 
    return np.dot(U[:,:k], np.dot(sigma_k, V[:k,:])) 


# In[44]:


#k=100
R_100 = reconstruct_rank_k_matrix(U, S, V, 100)
print("Prediction ratings for the movie ID 1377 by the 3 users are: {0} respectively.".format(R_100[[17,21,23],1376]))


# In[45]:


# k = 1000
R_1000 = reconstruct_rank_k_matrix(U, S, V, 1000)
print("Prediction ratings for the movie ID 1377 by the 3 users are: {0} respectively.".format(R_1000[[17,21,23],1376]))


# In[77]:


# k = 2000
R_2000 = reconstruct_rank_k_matrix(U, S, V, 2000)
print("Prediction ratings for the movie ID 1377 by the 3 users are: {0} respectively.".format(R_2000[[17,21,23],1376]))


# In[78]:


# k = 3000
R_3000 = reconstruct_rank_k_matrix(U, S, V, 3000)
print("Prediction ratings for the movie ID 1377 by the 3 users are: {0} respectively.".format(R_3000[[17,21,23],1376]))


# In[79]:


sample = data.loc[data['MovieID'] == 1377]
sample = sample.loc[sample.UserID.isin([18,22,24])]
print("The actual ratings given by 3 users for the movie:1377 are: ", sample.Ratings.values)


# ## Question 5

# ### Cosine Similarity
# Cosine similarity is a metric used to measure how similar two vectors are. Mathematically, it measures the cosine of the angle between two vectors projected in a multi-dimensional space. Cosine similarity is high if the angle between two vectors is 0, and the output value ranges within $cosine(x,y) \in [0,1]$. $0$ means there is no similarity (perpendicular), where $1$ (parallel) means that both the items are 100% similar.
# 
# $$ cosine(x,y) = \frac{x^T y}{||x|| ||y||}  $$

# **Based on the reconstruction rank-1000 rating matrix $R_{1000}$ and the cosine similarity,** sort the movies which are most similar. You will have a function `top_cosine_similarity` which sorts data by its similarity to a movie with ID `movie_id` and returns the top $n$ items, and a second function `print_similar_movies` which prints the titles of said similar movies. Return the top 5 movies for the movie with ID `1377` (*Batman Returns*):

# In[82]:


# Sort the movies based on cosine similarity
def top_cosine_similarity(data, movie_id, top_n=5):
    # Movie id starts from 1
    #Use the calculation formula above
    row = list(data[movie_id-1, :])
    movie_ids = sorted(range(len(row)), key=lambda k: row[k])[-1:-top_n-1:-1]
    return [i+1 for i in movie_ids]
def print_similar_movies(movie_data,movieID,top_indexes):
    similar_movies = list(movie_data.loc[movie_data.MovieID.isin(top_indexes)].Title.unique())
    print('Most Similar movies: ', similar_movies)


# In[83]:


result = np.dot(R_1000.T, R_1000)/(np.linalg.norm(R_1000.T)*np.linalg.norm(R_1000))


# In[84]:


# Print the top 5 movies for Batman Returns
movie_id = 1377
top_indexes = top_cosine_similarity(result, movie_id)
print_similar_movies(data, movie_id, top_indexes)

