__author__ = 'slyfocks'
import json
import numpy as np
import csv
import gender
#right now the code is agnostic to business data. coming shortly...
with open('yelp_test_set_user.json') as file:
    data = [json.loads(line) for line in file]
#convert names to all lowercase for alphabetizing purposes
id_names = {entry['user_id']: entry['name'].strip().lower() for entry in data}

with open('yelp_test_set_review.json') as file:
    review_data = [json.loads(line) for line in file]
user_ids = [entry['user_id'] for entry in review_data]


def names(user_ids):
    names = []
    for user_id in user_ids:
        try:
            names.append(id_names[user_id])
        except KeyError:
            names.append('unknown')
    return names


#returns list of genders
def genders(names):
    name_list = names
    return gender.name_gender(name_list)


def gender_means(gender_list):
    female_mean_stars = gender.training_mean('female')
    male_mean_stars = gender.training_mean('male')
    unknown_mean_stars = gender.training_mean('unknown')
    both_mean_stars = gender.training_mean('both')
    gender_means = []
    for index, id_gender in enumerate(gender_list):
        if id_gender == 'female':
            gender_means.append({'RecommendationId': index+1, 'Stars': female_mean_stars})
        elif id_gender == 'male':
            gender_means.append({'RecommendationId': index+1, 'Stars': male_mean_stars})
        elif id_gender == 'unknown':
            gender_means.append({'RecommendationId': index+1, 'Stars': unknown_mean_stars})
        else:
            gender_means.append({'RecommendationId': index+1, 'Stars': both_mean_stars})
    return gender_means


def main():
    name_list = names(user_ids)
    ratings_array = gender_means(genders(name_list))
    keys = ['RecommendationId', 'Stars']
    f = open('people.csv', 'w')
    dict_writer = csv.DictWriter(f, keys)
    dict_writer.writer.writerow(keys)
    dict_writer.writerows(ratings_array)

if __name__ == '__main__':
    main()