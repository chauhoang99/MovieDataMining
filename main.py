import csv
import json
import ast
import pandas
import pprint
import arff


def create_final_file():
    genreAndKeyword = pandas.read_csv("/home/hoangchau/study/data mining and text analysis/summative/Movie Data/Movie Data/genreAndKeyword.csv")
    data = dict()
    genreAndKeyword['genre'] = genreAndKeyword['genre'].apply(lambda x: x.replace(' ', '_'))
    genreAndKeyword['keyword'] = genreAndKeyword['keyword'].apply(lambda x: str(x).replace("'", ''))
    for index, row in genreAndKeyword.iterrows():
        if str(row['movieId']) + row['genre'] not in data.keys():
            data[str(row['movieId']) + row['genre']] = dict()
            data[str(row['movieId']) + row['genre']]['movieId'] = row['movieId']
            data[str(row['movieId']) + row['genre']]['genre'] = row['genre']
        if not data[str(row['movieId']) + row['genre']].get('keywords'):
            data[str(row['movieId']) + row['genre']]['keywords'] = list()
        data[str(row['movieId']) + row['genre']]['keywords'].append(row['keyword'])
    newData = list()
    for key, value in data.items():
        newDict = value
        newDict['itemId'] = key
        newData.append(newDict)
    df = pandas.DataFrame(newData)
    # Drama is an umbrella genre that cover many more specific genres in it.
    df = df[df.genre != 'Drama']
    df['keywords'] = df['keywords'].apply(lambda x: ','.join(x))
    # df2 = pandas.concat([df[['itemId', 'movieId', 'genre']], pandas.get_dummies(df['keywords'].apply(pandas.Series), prefix='keyword')], axis=1)
    df2 = pandas.concat([df[['itemId', 'movieId', 'genre']], df['keywords'].str.get_dummies(sep=',')], axis=1)
    columns = list(df2.columns)
    for i in columns:
        print("@attribute " + i.replace(' ', '_') + ' numeric')
    # df2.to_csv(path_or_buf='/home/hoangchau/study/data mining and text analysis/summative/Movie Data/Movie Data/GenreData/set2/train.csv', index=False)
    df2.to_csv(path_or_buf='/home/hoangchau/study/data mining and text analysis/summative/Movie Data/Movie Data/GenreData/set2/test.csv', index=False)


def inner_json_to_list(x):
    x = ast.literal_eval(x)
    result = list()
    if not x or not isinstance(x, list):
        return []
    for i in x:
        result.append(i.get('name'))
    return result


def clean_field_adult(x):
    if x == 'False':
        return 0
    else:
        return 1


def process_rating_data():
    df_from_file = pandas.read_csv("/home/hoangchau/study/data mining and text analysis/summative/Movie Data/Movie Data/movies_metadata.csv")
    movie_data = df_from_file[['id', 'adult', 'genres', 'original_language', 'production_countries', 'spoken_languages']]
    movie_data['adult'] = movie_data['adult'].apply(lambda x: clean_field_adult(x))

    movie_data['genres'] = movie_data['genres'].apply(lambda x: inner_json_to_list(x))
    movie_data['genres'] = movie_data['genres'].apply(lambda x: ','.join(x))

    movie_data['production_countries'] = movie_data['production_countries'].fillna('[]')
    movie_data['production_countries'] = movie_data['production_countries'].apply(lambda x: inner_json_to_list(x))
    movie_data['production_countries'] = movie_data['production_countries'].apply(lambda x: ','.join(x))

    movie_data['spoken_languages'] = movie_data['spoken_languages'].fillna('[]')
    movie_data['spoken_languages'] = movie_data['spoken_languages'].apply(lambda x: inner_json_to_list(x))
    movie_data['spoken_languages'] = movie_data['spoken_languages'].apply(lambda x: ','.join(x))

    movie_data['genres'] = movie_data['genres'].apply(lambda x: str(x).replace(' ', '_'))
    movie_data['production_countries'] = movie_data['production_countries'].apply(lambda x: str(x).replace(' ', '_'))
    movie_data['spoken_languages'] = movie_data['spoken_languages'].apply(lambda x: str(x).replace(' ', '_'))

    movie_data2 = pandas.concat([
        movie_data[['id', 'adult']],
        movie_data['genres'].str.get_dummies(sep=',').add_prefix('genre_'),
        movie_data['production_countries'].str.get_dummies(sep=',').add_prefix('production_country_'),
        movie_data['spoken_languages'].str.get_dummies(sep=',').add_prefix('spoken_language_'),
        movie_data['original_language'].str.get_dummies(sep=',').add_prefix('original_language_')
                                 ], axis=1)
    # columns = list(movie_data2.columns)
    # for i in columns:
    #     print("@attribute " + i.replace(' ', '_') + ' numeric')
    # movie_data2.to_csv(path_or_buf='/home/hoangchau/study/data mining and text analysis/summative/Movie Data/Movie Data/RecomendationData/set1/movies.csv', index=False)

    arff.dump(
        '/home/hoangchau/study/data mining and text analysis/summative/Movie Data/Movie Data/RecomendationData/set1/movies.arff',
        movie_data2.values,
        relation='movies',
        names=movie_data2.columns
    )

    rating = pandas.read_csv("/home/hoangchau/study/data mining and text analysis/summative/Movie Data/Movie Data/ratings.csv")
    rating = rating[(rating.userId == 1) | (rating.userId == 5829) | (rating.userId == 9173)]

    # rating.to_csv(path_or_buf='/home/hoangchau/study/data mining and text analysis/summative/Movie Data/Movie Data/RecomendationData/set1/ratings.csv', index=False)
    arff.dump(
        '/home/hoangchau/study/data mining and text analysis/summative/Movie Data/Movie Data/RecomendationData/set1/ratings.arff',
        rating.values,
        relation='ratings',
        names=rating.columns
    )


def handle_movie_id(x):
    try:
        return int(x)
    except ValueError:
        return x


def process_rating_data2():
    df_from_file = pandas.read_csv("/home/hoangchau/study/data mining and text analysis/summative/Movie Data/Movie Data/movies_metadata.csv")
    movie_data = df_from_file[['id', 'genres']]

    movie_data['genres'] = movie_data['genres'].apply(lambda x: inner_json_to_list(x))
    movie_data['genres'] = movie_data['genres'].apply(lambda x: ','.join(x))
    movie_data['id'] = movie_data['id'].apply(lambda x: handle_movie_id(x))
    movie_data = movie_data[movie_data['id'].apply(lambda x: isinstance(x, int))]

    rating = pandas.read_csv("/home/hoangchau/study/data mining and text analysis/summative/Movie Data/Movie Data/ratings.csv")
    rating = rating[['userId', 'movieId', 'rating']]
    rating1 = rating[rating.userId == 1]
    rating2 = rating[rating.userId == 5829]
    rating3 = rating[rating.userId == 9173]

    rating1 = rating1.merge(
        right=movie_data,
        how='inner',
        left_on='movieId',
        right_on='id',
    )
    rating1 = pandas.concat([
        rating1[['userId', 'movieId', 'rating']],
        rating1['genres'].str.get_dummies(sep=',').add_prefix('genre_'),
                                 ], axis=1)

    rating2 = rating2.merge(
        right=movie_data,
        how='inner',
        left_on='movieId',
        right_on='id',
    )
    rating2 = pandas.concat([
        rating2[['userId', 'movieId', 'rating']],
        rating2['genres'].str.get_dummies(sep=',').add_prefix('genre_'),
                                 ], axis=1)
    rating3 = rating3.merge(
        right=movie_data,
        how='inner',
        left_on='movieId',
        right_on='id',
    )
    rating3 = pandas.concat([
        rating3[['userId', 'movieId', 'rating']],
        rating3['genres'].str.get_dummies(sep=',').add_prefix('genre_'),
                                 ], axis=1)

    # rating.to_csv(path_or_buf='/home/hoangchau/study/data mining and text analysis/summative/Movie Data/Movie Data/RecomendationData/set1/ratings.csv', index=False)
    arff.dump(
        '/home/hoangchau/study/data mining and text analysis/summative/Movie Data/Movie Data/RecomendationData/set2/rating1.arff',
        rating1.values,
        relation='rating1',
        names=rating1.columns
    )
    arff.dump(
        '/home/hoangchau/study/data mining and text analysis/summative/Movie Data/Movie Data/RecomendationData/set2/rating2.arff',
        rating2.values,
        relation='rating2',
        names=rating2.columns
    )
    arff.dump(
        '/home/hoangchau/study/data mining and text analysis/summative/Movie Data/Movie Data/RecomendationData/set2/rating3.arff',
        rating3.values,
        relation='rating3',
        names=rating3.columns
    )


def process_rating_data3():
    credit_data = pandas.read_csv("/home/hoangchau/study/data mining and text analysis/summative/Movie Data/Movie Data/credits.csv")

    credit_data['cast'] = credit_data['cast'].apply(lambda x: inner_json_to_list(x))
    credit_data['cast'] = credit_data['cast'].apply(lambda x: [i.replace(' ', '_') for i in x])
    credit_data['cast'] = credit_data['cast'].apply(lambda x: [i.replace("'", '') for i in x])
    credit_data['cast'] = credit_data['cast'].apply(lambda x: ','.join(x))

    credit_data['crew'] = credit_data['crew'].apply(lambda x: inner_json_to_list(x))
    credit_data['crew'] = credit_data['crew'].apply(lambda x: [i.replace(' ', '_') for i in x])
    credit_data['crew'] = credit_data['crew'].apply(lambda x: [i.replace("'", '') for i in x])
    credit_data['crew'] = credit_data['crew'].apply(lambda x: ','.join(x))

    rating = pandas.read_csv("/home/hoangchau/study/data mining and text analysis/summative/Movie Data/Movie Data/ratings.csv")
    rating = rating[['userId', 'movieId', 'rating']]

    rating1 = rating[rating.userId == 1]
    rating1 = rating1.merge(
        right=credit_data,
        how='inner',
        left_on='movieId',
        right_on='id',
    )
    rating1 = pandas.concat([
        rating1[['userId', 'movieId', 'rating']],
        rating1['cast'].str.get_dummies(sep=',').add_prefix('cast_'),
        rating1['crew'].str.get_dummies(sep=',').add_prefix('crew_'),
    ], axis=1)
    arff.dump(
        '/home/hoangchau/study/data mining and text analysis/summative/Movie Data/Movie Data/RecomendationData/set3/rating1.arff',
        rating1.values,
        relation='rating1',
        names=rating1.columns
    )

    rating2 = rating[rating.userId == 5829]
    rating2 = rating2.merge(
        right=credit_data,
        how='inner',
        left_on='movieId',
        right_on='id',
    )
    rating2 = pandas.concat([
        rating2[['userId', 'movieId', 'rating']],
        rating2['cast'].str.get_dummies(sep=',').add_prefix('cast_'),
        rating2['crew'].str.get_dummies(sep=',').add_prefix('crew_'),
    ], axis=1)
    arff.dump(
        '/home/hoangchau/study/data mining and text analysis/summative/Movie Data/Movie Data/RecomendationData/set3/rating2.arff',
        rating2.values,
        relation='rating2',
        names=rating2.columns
    )

    rating3 = rating[rating.userId == 9173]
    rating3 = rating3.merge(
        right=credit_data,
        how='inner',
        left_on='movieId',
        right_on='id',
    )
    rating3 = pandas.concat([
        rating3[['userId', 'movieId', 'rating']],
        rating3['cast'].str.get_dummies(sep=',').add_prefix('cast_'),
        rating3['crew'].str.get_dummies(sep=',').add_prefix('crew_'),
    ], axis=1)
    arff.dump(
        '/home/hoangchau/study/data mining and text analysis/summative/Movie Data/Movie Data/RecomendationData/set3/rating3.arff',
        rating3.values,
        relation='rating3',
        names=rating3.columns
    )


if __name__ == '__main__':
    # create_final_file()
    process_rating_data3()
