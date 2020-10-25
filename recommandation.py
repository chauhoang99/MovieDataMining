from scipy import spatial
import pandas
import ast
import arff


class Recommender(object):
    user_rating_data = None
    credit_data = None
    movie_data = None
    rated_movies = None
    features = None
    user_id = None

    def __init__(self, user_id, user_rating_data, credit_data, number_of_recommendations):
        self.user_id = user_id
        self.user_rating_data = user_rating_data[user_rating_data['rating'] >= 3]
        self.credit_data = credit_data
        self.number_of_recommendation = number_of_recommendations
        self.query_relevant_movie_raw_data()
        self.recommend_products()

    def inner_json_to_list(self, x):
        x = ast.literal_eval(x)
        result = list()
        if not x or not isinstance(x, list):
            return []
        for i in x:
            result.append(i.get('name'))
        return result

    def query_relevant_movie_raw_data(self):
        rated_movies = self.user_rating_data.merge(
            right=self.credit_data,
            how='inner',
            left_on='movieId',
            right_on='id',
        )
        rated_movies['cast'] = rated_movies['cast'].apply(lambda x: self.inner_json_to_list(x))
        rated_movies['cast'] = rated_movies['cast'].apply(lambda x: [i.replace(' ', '_') for i in x])
        rated_movies['cast'] = rated_movies['cast'].apply(lambda x: [i.replace("'", '') for i in x])
        rated_movies['cast'] = rated_movies['cast'].apply(lambda x: [i.replace(".", '') for i in x])

        rated_movies['crew'] = rated_movies['crew'].apply(lambda x: self.inner_json_to_list(x))
        rated_movies['crew'] = rated_movies['crew'].apply(lambda x: [i.replace(' ', '_') for i in x])
        rated_movies['crew'] = rated_movies['crew'].apply(lambda x: [i.replace("'", '') for i in x])
        rated_movies['crew'] = rated_movies['crew'].apply(lambda x: [i.replace(".", '') for i in x])


        genre_data = pandas.read_csv(
            "/home/hoangchau/study/data mining and text analysis/summative/Movie Data/Movie Data/genreOfMovies.csv", delimiter=";"
        )
        genre_data['genres'] = genre_data['genres'].apply(lambda x: self.inner_json_to_list(x))
        genre_data['genres'] = genre_data['genres'].apply(lambda x: [i.replace(' ', '_') for i in x])
        genre_data['genres'] = genre_data['genres'].apply(lambda x: [i.replace("'", '') for i in x])
        genre_data['genres'] = genre_data['genres'].apply(lambda x: [i.replace(".", '') for i in x])
        genre_data['id'] = genre_data['id'].apply(lambda x: int(x))

        cast = rated_movies['cast'].to_list()
        crew = rated_movies['crew'].to_list()
        genre = genre_data['genres'].to_list()
        features = list()
        for names in cast + crew + genre:
            for i in names:
                features.append(i)
        features = list(set(features))

        rated_movies = rated_movies.merge(
            right=genre_data,
            how='left',
            left_on='id',
            right_on='id',
        )
        self.user_rating_data = rated_movies

        self.movie_data = self.credit_data[~self.credit_data['id'].isin(self.user_rating_data['movieId'])]
        self.credit_data = None
        self.user_rating_data = self.user_rating_data.sort_values('rating', ascending=False)
        self.user_rating_data.reset_index(inplace=True)
        del self.user_rating_data['index']
        del self.user_rating_data['movieId']

        self.movie_data['cast'] = self.movie_data['cast'].apply(lambda x: self.inner_json_to_list(x))
        self.movie_data['cast'] = self.movie_data['cast'].apply(lambda x: [i.replace(' ', '_') for i in x])
        self.movie_data['cast'] = self.movie_data['cast'].apply(lambda x: [i.replace("'", '') for i in x])
        self.movie_data['cast'] = self.movie_data['cast'].apply(lambda x: [i.replace(".", '') for i in x])

        self.movie_data['crew'] = self.movie_data['crew'].apply(lambda x: self.inner_json_to_list(x))
        self.movie_data['crew'] = self.movie_data['crew'].apply(lambda x: [i.replace(' ', '_') for i in x])
        self.movie_data['crew'] = self.movie_data['crew'].apply(lambda x: [i.replace("'", '') for i in x])
        self.movie_data['crew'] = self.movie_data['crew'].apply(lambda x: [i.replace(".", '') for i in x])
        self.movie_data = self.movie_data.merge(
            right=genre_data,
            how='left',
            left_on='id',
            right_on='id',
        )

        self.movie_data = self.movie_data[self.movie_data['cast'].apply(lambda x: any([i in features for i in x])) | self.movie_data['crew'].apply(lambda x: any([i in features for i in x]))]
        for feature in features:
            self.movie_data[feature] = self.movie_data.apply(lambda x: self.generate_binary_feature(feature, x), axis=1)
        del self.movie_data['cast']
        del self.movie_data['crew']
        del self.movie_data['genres']
        for feature in features:
            self.user_rating_data[feature] = self.user_rating_data.apply(lambda x: self.generate_binary_feature(feature, x), axis=1)
        del self.user_rating_data['cast']
        del self.user_rating_data['crew']
        del self.user_rating_data['genres']
        self.movie_data.reindex(sorted(self.movie_data.columns), axis=1)
        self.user_rating_data.reindex(sorted(self.user_rating_data.columns), axis=1)

        self.features = features

    def generate_binary_feature(self, value, row):
        if value in row['cast'] + row['crew'] + row['genres']:
            return 1
        else:
            return 0

    def recommend_products(self):
        top1 = self.user_rating_data.iloc[[0]].copy()
        del top1['userId']
        del top1['rating']
        del top1['id']
        top2 = self.user_rating_data.iloc[[1]].copy()
        del top2['userId']
        del top2['rating']
        del top2['id']
        top3 = self.user_rating_data.iloc[[2]].copy()
        del top3['userId']
        del top3['rating']
        del top3['id']
        movie_data = self.movie_data.copy()
        movie_data['cosine_distance'] = movie_data[self.features].apply(lambda x: self.cosine_distance(x, top1, 0), axis=1)
        movie_data = movie_data.sort_values('cosine_distance')
        to_recommand = movie_data.head(3)

        movie_data = self.movie_data.copy()
        movie_data['cosine_distance'] = movie_data[self.features].apply(lambda x: self.cosine_distance(x, top2, 1), axis=1)
        movie_data = movie_data.sort_values('cosine_distance')
        to_recommand = to_recommand.append(movie_data.head(3))

        movie_data = self.movie_data.copy()
        movie_data['cosine_distance'] = movie_data[self.features].apply(lambda x: self.cosine_distance(x, top3, 2), axis=1)
        movie_data = movie_data.sort_values('cosine_distance')
        to_recommand = to_recommand.append(movie_data.head(3))
        to_recommand.sort_values('cosine_distance')
        to_recommand.reset_index(inplace=True)
        to_recommand = to_recommand.head(int(self.number_of_recommendation))
        to_recommand = to_recommand.loc[:, (to_recommand != 0).any(axis=0)]
        del to_recommand['index']
        print("{} movies to recommend to user {}:".format(self.number_of_recommendation, self.user_id))
        print(to_recommand)
        print('Top 3 rated movies by user are: ')
        top_three = self.user_rating_data.head(3)
        top_three = top_three.loc[:, (top_three != 0).any(axis=0)]
        print(top_three)

    def cosine_distance(self, x, top, n):
        x = x.values.tolist()
        top = top.loc[n, :].values.tolist()
        return spatial.distance.cosine(x, top)


class Recommender2(object):
    user_rating_data = None
    user_id = None

    def __init__(self, user_id, user_rating_data):
        self.user_id = user_id
        self.user_rating_data = user_rating_data
        self.query_relevant_movie_raw_data()

    def inner_json_to_list(self, x):
        x = ast.literal_eval(x)
        result = list()
        if not x or not isinstance(x, list):
            return []
        for i in x:
            result.append(i.get('name'))
        return result

    def query_relevant_movie_raw_data(self):
        credit_data = pandas.read_csv(
            "/home/hoangchau/study/data mining and text analysis/summative/Movie Data/Movie Data/credits.csv")

        self.user_rating_data = self.user_rating_data.merge(
            right=credit_data,
            how='inner',
            left_on='movieId',
            right_on='id',
        )
        self.user_rating_data['like'] = self.user_rating_data['rating'].apply(lambda x: 1 if x >= 4 else 0)
        self.user_rating_data['cast'] = self.user_rating_data['cast'].apply(lambda x: self.inner_json_to_list(x))
        self.user_rating_data['cast'] = self.user_rating_data['cast'].apply(lambda x: [i.replace(' ', '_') for i in x])
        self.user_rating_data['cast'] = self.user_rating_data['cast'].apply(lambda x: [i.replace("'", '') for i in x])
        self.user_rating_data['cast'] = self.user_rating_data['cast'].apply(lambda x: [i.replace(".", '') for i in x])
        # self.user_rating_data['cast'] = self.user_rating_data['cast'].apply(lambda x: ','.join(x))

        self.user_rating_data['crew'] = self.user_rating_data['crew'].apply(lambda x: self.inner_json_to_list(x))
        self.user_rating_data['crew'] = self.user_rating_data['crew'].apply(lambda x: [i.replace(' ', '_') for i in x])
        self.user_rating_data['crew'] = self.user_rating_data['crew'].apply(lambda x: [i.replace("'", '') for i in x])
        self.user_rating_data['crew'] = self.user_rating_data['crew'].apply(lambda x: [i.replace(".", '') for i in x])
        # self.user_rating_data['crew'] = self.user_rating_data['crew'].apply(lambda x: ','.join(x))

        genre_data = pandas.read_csv(
            "/home/hoangchau/study/data mining and text analysis/summative/Movie Data/Movie Data/genreOfMovies.csv",
            delimiter=";"
        )
        genre_data['genres'] = genre_data['genres'].apply(lambda x: self.inner_json_to_list(x))
        genre_data['genres'] = genre_data['genres'].apply(lambda x: [i.replace(' ', '_') for i in x])
        genre_data['genres'] = genre_data['genres'].apply(lambda x: [i.replace("'", '') for i in x])
        genre_data['genres'] = genre_data['genres'].apply(lambda x: [i.replace(".", '') for i in x])
        # genre_data['genres'] = genre_data['genres'].apply(lambda x: ','.join(x))

        genre_data['id'] = genre_data['id'].apply(lambda x: int(x))

        # cast = rated_movies['cast'].to_list()
        # crew = rated_movies['crew'].to_list()
        # genre = genre_data['genres'].to_list()
        # features = list()
        # for names in cast + crew + genre:
        #     for i in names:
        #         features.append(i)
        # features = list(set(features))

        self.user_rating_data = self.user_rating_data.merge(
            right=genre_data,
            how='left',
            left_on='id',
            right_on='id',
        )
        del self.user_rating_data['movieId']

        self.user_rating_data['features'] = self.user_rating_data[['cast', 'crew', 'genres']].apply(lambda x: x['cast'] + x['crew'] + x['genres'], axis=1)
        del self.user_rating_data['cast']
        del self.user_rating_data['crew']
        del self.user_rating_data['genres']
        self.user_rating_data['features'] = self.user_rating_data['features'].apply(lambda x: ','.join(x))

        # self.user_rating_data = self.user_rating_data.sample(frac=0.2).reset_index(drop=True)
        # print(self.user_rating_data.count() + 1)
        #
        # features = list()
        # for x in self.user_rating_data['features'].to_list():
        #     for i in x:
        #         features.append(i)
        # for feature in features:
        #     self.user_rating_data[feature] = self.user_rating_data.apply(lambda x: self.generate_binary_feature(feature, x), axis=1)

        # del self.user_rating_data['features']
        self.user_rating_data = pandas.concat([
            self.user_rating_data[['userId', 'id', 'like']],
            self.user_rating_data['features'].str.get_dummies(sep=',')
        ],
            axis=1
        )
        # arff.dump(
        # '/home/hoangchau/study/data mining and text analysis/summative/Movie Data/Movie Data/RecomendationData/set4/user{}(4).arff'.format(self.user_id),
        # self.user_rating_data.values,
        # relation='movies',
        # names=self.user_rating_data.columns
        # )

        movie_data = credit_data[~credit_data['id'].isin(self.user_rating_data['id'])]
        movie_data = movie_data.merge(
            right=genre_data,
            how='left',
            left_on='id',
            right_on='id',
        )

        movie_data['like'] = '?'
        movie_data['cast'] = movie_data['cast'].apply(lambda x: self.inner_json_to_list(x))
        movie_data['cast'] = movie_data['cast'].apply(lambda x: [i.replace(' ', '_') for i in x])
        movie_data['cast'] = movie_data['cast'].apply(lambda x: [i.replace("'", '') for i in x])
        movie_data['cast'] = movie_data['cast'].apply(lambda x: [i.replace(".", '') for i in x])
        movie_data['cast'] = movie_data['cast'].apply(lambda x: self.remove_unnecessary_feature(x, self.user_rating_data.columns))

        movie_data['crew'] = movie_data['crew'].apply(lambda x: self.inner_json_to_list(x))
        movie_data['crew'] = movie_data['crew'].apply(lambda x: [i.replace(' ', '_') for i in x])
        movie_data['crew'] = movie_data['crew'].apply(lambda x: [i.replace("'", '') for i in x])
        movie_data['crew'] = movie_data['crew'].apply(lambda x: [i.replace(".", '') for i in x])
        movie_data['crew'] = movie_data['crew'].apply(lambda x: self.remove_unnecessary_feature(x, self.user_rating_data.columns))

        movie_data['features'] = movie_data[['cast', 'crew', 'genres']].apply(lambda x: x['cast'] + x['crew'] + x['genres'], axis=1)

        del movie_data['cast']
        del movie_data['crew']
        del movie_data['genres']
        movie_data['features'] = movie_data['features'].apply(lambda x: ','.join(x))
        movie_data = pandas.concat([
            movie_data[['id', 'like']],
            movie_data['features'].str.get_dummies(sep=',')
        ],
            axis=1
        )
        arff.dump(
        '/home/hoangchau/study/data mining and text analysis/summative/Movie Data/Movie Data/RecomendationData/set4/moviesNotRatedUser{}.arff'.format(self.user_id),
        movie_data.values,
        relation='unrated_movies',
        names=movie_data.columns
        )

    def remove_unnecessary_feature(self, x, features):
        new_list =list()
        for i in x:
            if i in features:
                new_list.append(i)
        return new_list

    def generate_binary_feature(self, value, row):
        if value in row['features']:
            return 1
        else:
            return 0


# if __name__ == '__main__':
#     # user_id = input("User ID: ")
#     # number_of_recommendations = input("Number of recommendations: ")
#     # rating_data = pandas.read_csv(
#     #     "/home/hoangchau/study/data mining and text analysis/summative/Movie Data/Movie Data/ratings.csv")
#     # rating_data = rating_data[['userId', 'movieId', 'rating']]
#     # rating_data = rating_data[rating_data.userId == int(user_id)]
#     # credit_data = pandas.read_csv("/home/hoangchau/study/data mining and text analysis/summative/Movie Data/Movie Data/credits.csv")
#     #
#     # recommender = Recommender(user_id, rating_data, credit_data, int(number_of_recommendations))


if __name__ == '__main__':
    user_id = input("User ID: ")
    rating_data = pandas.read_csv(
        "/home/hoangchau/study/data mining and text analysis/summative/Movie Data/Movie Data/ratings.csv")
    rating_data = rating_data[['userId', 'movieId', 'rating']]
    rating_data = rating_data[rating_data.userId == int(user_id)]
    recommender = Recommender2(user_id, rating_data)
