import csv
import pandas as pd
import numpy as np
from joblib import dump, load
from scipy.sparse import csr_matrix
from pandas.api.types import CategoricalDtype
from joblib import dump, load



# read the csv data file
df = pd.read_csv('static/bgg-13m-reviews.csv',index_col=0)
#data set with scores
# select particular columns
bgg_r = df[["user","rating","ID",'name']]
#Add year to our ds
ds_score = pd.read_csv('static/2019-05-02.csv',index_col=0)
score = ds_score[['Name', "Year","URL"]]
#We remove old versions of games
url = score.groupby("Name")['URL'].apply(list).reset_index(name='URL')
score = score.groupby("Name")['Year'].apply(list).reset_index(name='Year')
score = score.merge(url,left_on='Name',right_on="Name")
score["Year"] = score["Year"].apply(max)
score = score.rename(columns={"Name": "name"})
bgg_r = bgg_r.merge(score,left_on='name',right_on="name")
#We add average rating
g = bgg_r.groupby('ID', as_index=False)['rating'].mean()
g = g.rename(columns={"rating": "Bayes average"})
bgg_r = bgg_r.merge(g,left_on='ID',right_on="ID")
# rename game ID column
bgg_r = bgg_r.rename(columns={'ID': 'gameId'})
#extra filtered
bgg_r2 = bgg_r[bgg_r.groupby("gameId")['gameId'].transform('size') > 200]
#removing old games (usualy they not available for purchase)
bgg_r2 = bgg_r2[bgg_r["Year"]>1995]
#We setting minimal threshold for user
bgg_r2 = bgg_r2[bgg_r.groupby("user")['user'].transform('size') > 10]
bgg_r2 = bgg_r2[bgg_r.groupby("user")['Bayes average'].transform('size') > 4]
# drop rows with null values
bgg_r2 = bgg_r2.dropna()
# create new ids for users and games (this is used for building the sparse matrix)
user_c2 = pd.Categorical(bgg_r2.user)
game_c2 = pd.Categorical(bgg_r2.gameId)

# add new id rows
bgg_r2['userIdx'] = user_c2.codes
bgg_r2['gameIdx'] = game_c2.codes

n_games2 = game_c2.categories.size
n_users2 = user_c2.categories.size
# build user-item sparse matrix
ui_sparse_matrix2 = csr_matrix((bgg_r2["rating"], (bgg_r2['userIdx'], bgg_r2['gameIdx'])), shape=(n_users2, n_games2))

# build dataframe from user-item sparse matrix
df_ui_matrix2 = pd.DataFrame.sparse.from_spmatrix(ui_sparse_matrix2)


df_games = bgg_r2[['gameId','name']].drop_duplicates()
df_ratings = bgg_r2[['userIdx','gameId','Bayes average']].drop_duplicates()

R = df_ui_matrix2.values
user_ratings_mean = np.mean(R, axis = 1)
R = R - user_ratings_mean.reshape(-1, 1)
from scipy.sparse.linalg import svds
U, sigma, Vt = svds(R, k = 100)
sigma = np.diag(sigma)
all_user_predicted_ratings = np.dot(np.dot(U, sigma), Vt) + user_ratings_mean.reshape(-1, 1)
preds_df = pd.DataFrame(all_user_predicted_ratings, columns = df_ui_matrix2.columns)

dump(preds_df, 'preds_df.pkl')
dump(df_games, 'df_games.pkl')
dump(df_ratings, 'df_ratings.pkl')
dump(bgg_r2, 'bgg_r2.pkl')