__author__ = 'slyfocks'
import json
import numpy as np
import csv
import matplotlib.pyplot as plt
import gender
import review_parse
import business_analysis as banal
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


def gender_means_recommendation(gender_list):
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


#takes user_id:stars and user_id:grade_level and creates stars:grade_level
def stars_grades():
    id_grades = review_parse.id_grades()
    id_star = gender.id_stars()
    grades_keys = id_grades.keys()
    star_keys = id_star.keys()
    return [(id_grades[id], id_star[id]) for id in user_ids if id in (grades_keys and star_keys)]


def stars_grade_lists():
    stars_grades_tuples = stars_grades()
    return [list(t) for t in zip(*stars_grades_tuples)]


def reviews_grades():
    id_grades = review_parse.id_grades()
    id_review = gender.id_reviews()
    grades_keys = id_grades.keys()
    review_keys = id_review.keys()
    return [(id_grades[id], id_review[id]) for id in user_ids if id in (grades_keys and review_keys)]


def reviews_grade_lists():
    reviews_grades_tuples = reviews_grades()
    return [list(t) for t in zip(*reviews_grades_tuples)]


#star rating and grade_level for a particular review
def grade_star_lists():
    grades = review_parse.grade_stars().keys()
    stars = list(review_parse.grade_stars().values())
    return [grades, stars]


def main():
    name_list = names(user_ids)
    business_ratings = banal.predicted_business_rating()
    gender_ratings = gender_means_recommendation(genders(name_list))
    business_ids = [entry['business_id'] for entry in review_data]
    for i in range(len(review_data)):
        try:
            business_rating = business_ratings[business_ids[i]]
        except KeyError:
            business_rating = gender_ratings[i]['Stars']
        gender_rating = gender_ratings[i]['Stars']
        overall_rating = (business_rating + gender_rating)/2
        gender_ratings[i]['Stars'] = overall_rating
    keys = ['RecommendationId', 'Stars']
    f = open('businesspeople.csv', 'w')
    dict_writer = csv.DictWriter(f, keys)
    dict_writer.writer.writerow(keys)
    dict_writer.writerows(gender_ratings)

if __name__ == '__main__':
    main()