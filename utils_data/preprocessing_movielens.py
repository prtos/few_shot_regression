import os
import pandas as pd
import numpy as np
from scipy import stats
from collections import Counter
from scipy.sparse.linalg import svds
import pickle


MOVIELENS_FOLDER = "../datasets/ml-20m/"
OUT_FOLDER = "../datasets/movielens/"
MOVIES_FEATURES_FILE = os.path.join(OUT_FOLDER, 'movies_features.pkl')
RATING_FILENAME = os.path.join(MOVIELENS_FOLDER, "ratings.csv")
MAX_USERS, MAX_MOVIES = 6000, 4000
NB_FEATURES = 100

# def get_latest_data():
#     download('')


def latent_factor_decomposition(data, saving_file=MOVIES_FEATURES_FILE):
    ratings = data.pivot(index='userId', columns='movieId', values='rating').fillna(0)
    movies = ratings.columns.values
    ratings = ratings.as_matrix()
    user_ratings_mean = np.mean(ratings, axis=1)
    rating_demeaned = ratings - user_ratings_mean.reshape(-1, 1)
    U, sigma, Vt = svds(rating_demeaned, k=NB_FEATURES)
    Vt = Vt.transpose()
    movies_features = dict(zip(movies, Vt))
    with open(saving_file, 'wb') as f:
        pickle.dump(movies_features, f, pickle.HIGHEST_PROTOCOL)


def create_few_shot_dataset():
    os.makedirs(OUT_FOLDER, exist_ok=True)
    data = pd.read_csv(RATING_FILENAME)
    counts_movies_per_user = data.groupby(['userId'], as_index=False).size().reset_index(name='counts')
    counts_movies_per_user = counts_movies_per_user.sort_values('counts', ascending=False)
    users_of_interest = counts_movies_per_user.head(n=MAX_USERS).userId.tolist()

    counts_users_per_movie = data.groupby(['movieId'], as_index=False).size().reset_index(name='counts')
    counts_users_per_movie = counts_users_per_movie.sort_values('counts', ascending=False)
    movies_of_interest = counts_users_per_movie.head(n=MAX_MOVIES).movieId.tolist()

    data = data[data.userId.isin(users_of_interest)]
    data = data[data.movieId.isin(movies_of_interest)]
    latent_factor_decomposition(data)

    for user in users_of_interest:
        temp = data[data.userId == user]
        temp = temp[['movieId', 'rating']]
        print(temp.shape)
        temp.to_csv(os.path.join(OUT_FOLDER, 'user{}.txt'.format(user)),
                    sep="\t", index=False, header=False)


if __name__ == '__main__':
    create_few_shot_dataset()