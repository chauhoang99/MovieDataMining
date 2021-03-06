from scipy import spatial
import pandas
import ast


class Recommender(object):
    user_rating_data = None
    credit_data = None
    movie_data = None
    rated_movies = None
    features = None

    def __init__(self, user_rating_data, credit_data, number_of_recommendations):
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

        keyword_data = pandas.read_csv(
            "/home/hoangchau/study/data mining and text analysis/summative/Movie Data/Movie Data/keywords.csv"
        )
        keyword_data['keywords'] = keyword_data['keywords'].apply(lambda x: self.inner_json_to_list(x))
        keyword_data['keywords'] = keyword_data['keywords'].apply(lambda x: [i.replace(' ', '_') for i in x])
        keyword_data['keywords'] = keyword_data['keywords'].apply(lambda x: [i.replace("'", '') for i in x])
        keyword_data['keywords'] = keyword_data['keywords'].apply(lambda x: [i.replace(".", '') for i in x])
        # keyword_data['id'] = keyword_data['id'].apply(lambda x: int(x))

        cast = rated_movies['cast'].to_list()
        crew = rated_movies['crew'].to_list()
        keyword = keyword_data['keywords'].to_list()
        features = list()
        for names in cast + crew + keyword:
            for i in names:
                features.append(i)
        features = list(set(features))

        rated_movies = rated_movies.merge(
            right=keyword_data,
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
            right=keyword_data,
            how='left',
            left_on='id',
            right_on='id',
        )

        self.movie_data = self.movie_data[self.movie_data['cast'].apply(lambda x: any([i in features for i in x])) | self.movie_data['crew'].apply(lambda x: any([i in features for i in x]))]
        for feature in features:
            self.movie_data[feature] = self.movie_data.apply(lambda x: self.generate_binary_feature(feature, x), axis=1)
        del self.movie_data['cast']
        del self.movie_data['crew']
        del self.movie_data['keywords']
        for feature in features:
            self.user_rating_data[feature] = self.user_rating_data.apply(lambda x: self.generate_binary_feature(feature, x), axis=1)
        del self.user_rating_data['cast']
        del self.user_rating_data['crew']
        del self.user_rating_data['keywords']

        self.features = features

    def generate_binary_feature(self, value, row):
        if value in row['cast'] + row['crew'] + row['keywords']:
            return 1
        else:
            return 0

    def recommend_products(self):
        print('Top 3 rated movies by user are: '.format(self.number_of_recommendation))
        top1 = self.user_rating_data.iloc[[0]].copy()
        top1 = top1.reindex(sorted(top1.columns), axis=1)
        print(top1['id'])
        del top1['userId']
        del top1['rating']
        del top1['id']
        top2 = self.user_rating_data.iloc[[1]].copy()
        top2 = top2.reindex(sorted(top2.columns), axis=1)
        print(top2['id'])
        del top2['userId']
        del top2['rating']
        del top2['id']
        top3 = self.user_rating_data.iloc[[2]].copy()
        top3 = top3.reindex(sorted(top3.columns), axis=1)
        print(top3['id'])
        del top3['userId']
        del top3['rating']
        del top3['id']
        movie_data = self.movie_data.copy()
        movie_data = movie_data.reindex(sorted(movie_data.columns), axis=1)
        movie_data['cosine_distance'] = movie_data[self.features].apply(lambda x: self.cosine_distance(x, top1), axis=1)
        movie_data = movie_data.sort_values('cosine_distance')
        print(movie_data.head(3))
        # movie_data.reset_index(inplace=True)
        # del movie_data['index']

    def cosine_distance(self, x, top):
        x = x.values.tolist()
        top = top.loc[0, :].values.tolist()
        return spatial.distance.cosine(x, top)

if __name__ == '__main__':
    user_id = input("User ID: ")
    number_of_recommendations = input("Number of recommendations: ")
    rating_data = pandas.read_csv(
        "/home/hoangchau/study/data mining and text analysis/summative/Movie Data/Movie Data/ratings.csv")
    rating_data = rating_data[['userId', 'movieId', 'rating']]
    rating_data = rating_data[rating_data.userId == int(user_id)]
    credit_data = pandas.read_csv("/home/hoangchau/study/data mining and text analysis/summative/Movie Data/Movie Data/credits.csv")

    recommender = Recommender(rating_data, credit_data, int(number_of_recommendations))
