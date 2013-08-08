__author__ = 'slyfocks'
import json
import numpy as np
import csv
import matplotlib.pyplot as plt
import gender
import review_parse as rp
import business_analysis as banal
import funny_useful_cool as fuc
import user_analysis as ua

with open('yelp_test_set_user.json') as file:
    data = [json.loads(line) for line in file]
#convert names to all lowercase for alphabetizing purposes
id_names = {entry['user_id']: entry['name'].strip().lower() for entry in data}

with open('yelp_test_set_review.json') as file:
    review_data = [json.loads(line) for line in file]
user_ids = [entry['user_id'] for entry in review_data]
business_ids = [entry['business_id'] for entry in review_data]

with open('yelp_training_set_user.json') as file:
    training_data = [json.loads(user) for user in file]


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


def id_stars():
    return {member['user_id']: member['average_stars'] for member in training_data}


def id_review_dict():
    return {member['user_id']: member['review_count'] for member in training_data}


#takes user_id:stars and user_id:grade_level and creates stars:grade_level
def stars_grades():
    id_grades = rp.id_grades()
    id_star = gender.id_stars()
    grades_keys = list(id_grades.keys())
    star_keys = list(id_star.keys())
    return [(id_grades[id], id_star[id]) for id in user_ids if id in set(grades_keys).intersection(star_keys)]


def stars_grade_lists():
    stars_grades_tuples = stars_grades()
    return [list(t) for t in zip(*stars_grades_tuples)]


def reviews_grades():
    id_grades = rp.id_grades()
    id_review = gender.id_reviews()
    grades_keys = list(id_grades.keys())
    review_keys = list(id_review.keys())
    return [(id_grades[id], id_review[id]) for id in user_ids if id in set(grades_keys).intersection(review_keys)]


def reviews_grade_lists():
    reviews_grades_tuples = reviews_grades()
    return [list(t) for t in zip(*reviews_grades_tuples)]


#star rating and grade_level for a particular review
def grade_star_lists():
    grades = list(rp.grade_stars().keys())
    stars = list(rp.grade_stars().values())
    return [grades, stars]


def test_id_reviews():
    return {member['user_id']: member['review_count'] for member in data}


def main():
    #user_ids is test set_review_ids
    gender_ratings = gender.id_gender()
    test_gender_ratings = gender.test_id_gender()
    female_mean = gender.training_mean('female')
    male_mean = gender.training_mean('male')
    unknown_mean = gender.training_mean('unknown')
    both_mean = gender.training_mean('both')
    mean_stars = fuc.mean_user_stars()

    #for users in training user set
    id_stars_dict = id_stars()
    id_reviews = id_review_dict()

    #user_id stuff
    test_users = ua.review_test_users()
    training_users = ua.review_training_users()
    all_groups = ua.all_group_users()

    #these variables will be for users who have writing samples available
    parse_review_users = ua.review_training_test_users()
    parse_avg = rp.id_grade_avg()
    review_user_stars = rp.id_stars()
    review_stars_average = rp.id_stars_avg()


    #business stuff
    test_businesses = banal.review_test_businesses()
    training_businesses = banal.review_training_businesses()
    training_review_businesses = list(banal.id_stars().keys())
    #test_categories = [banal.categories(entry) for entry in test_businesses]
    #training_categories = [banal.categories(entry) for entry in training_businesses]
    expected_business_rating = banal.predicted_business_rating()

    #funny_useful_cool stuff, for training users only!
    fuc_rating_dict = fuc.predicted_rating()
    total_fuc_ratings = fuc.total_fuc()

    #where final ratings go for users
    user_ratings = {}

    #where final ratings go for businesses
    business_ratings = {}

    #in case all of these loops don't contain certain users, initialize all users to the mean
    for user in user_ids:
        user_ratings[user] = mean_stars

    for user in training_users:
        if gender_ratings[user] == 'female':
            user_gender_rating = female_mean
        elif gender_ratings[user] == 'male':
            user_gender_rating = male_mean
        elif gender_ratings[user] == 'unknown':
            user_gender_rating = unknown_mean
        else:
            user_gender_rating = both_mean
        user_stars = id_stars_dict[user]
        review_count = id_reviews[user]
        #fuc rating given on 1-5 scale based on lms regression on funny, useful, cool ratings and star ratings
        fuc_rating = fuc_rating_dict[user]
        fuc_count = total_fuc_ratings[user]
        rating = (np.log(review_count)*user_stars + user_gender_rating
                  + fuc_rating*np.log(fuc_count))/(np.log(review_count) + np.log(fuc_count) + 1)
        user_ratings[user] = rating

    for user in test_users:
        if test_gender_ratings[user] == 'female':
            user_gender_rating = female_mean
        elif test_gender_ratings[user] == 'male':
            user_gender_rating = male_mean
        elif test_gender_ratings[user] == 'unknown':
            user_gender_rating = unknown_mean
        else:
            user_gender_rating = both_mean
        rating = user_gender_rating
        user_ratings[user] = rating

    for user in parse_review_users:
        try:
            user_review_rating = parse_avg[user]
        except KeyError:
            user_review_rating = mean_stars
        user_stars = review_stars_average[user]
        review_count = len(review_user_stars[user])
        rating = (user_review_rating + user_stars*np.log(review_count))/(1 + np.log(review_count))
        user_ratings[user] = rating

    for user in set(test_users).intersection(training_users):
        if gender_ratings[user] == 'female':
            user_gender_rating = female_mean
        elif gender_ratings[user] == 'male':
            user_gender_rating = male_mean
        elif gender_ratings[user] == 'unknown':
            user_gender_rating = unknown_mean
        else:
            user_gender_rating = both_mean
        user_stars = id_stars_dict[user]
        review_count = id_reviews[user]
        fuc_rating = fuc_rating_dict[user]
        fuc_count = total_fuc_ratings[user]
        rating = (np.log(review_count)*user_stars + user_gender_rating
                  + fuc_rating*np.log(fuc_count))/(np.log(review_count) + np.log(fuc_count) + 1)
        user_ratings[user] = rating

    for user in set(parse_review_users).intersection(training_users):
        if gender_ratings[user] == 'female':
            user_gender_rating = female_mean
        elif gender_ratings[user] == 'male':
            user_gender_rating = male_mean
        elif gender_ratings[user] == 'unknown':
            user_gender_rating = unknown_mean
        else:
            user_gender_rating = both_mean
        try:
            user_review_rating = parse_avg[user]
        except KeyError:
            user_review_rating = mean_stars
        user_stars = id_stars_dict[user]
        user_stars_review = review_stars_average[user]
        review_count = id_reviews[user]
        review_count_reviews = len(review_user_stars[user])
        fuc_rating = fuc_rating_dict[user]
        fuc_count = total_fuc_ratings[user]
        rating = ((np.log(review_count_reviews)*user_review_rating + user_stars*np.log(review_count)
                  + user_stars_review*np.log(review_count_reviews) + user_gender_rating
                  + fuc_rating*np.log(fuc_count))/(2*np.log(review_count_reviews)
                                                   + np.log(review_count)
                                                   + np.log(fuc_count) + 1))
        user_ratings[user] = rating

    for user in set(parse_review_users).intersection(test_users):
        if test_gender_ratings[user] == 'female':
            user_gender_rating = female_mean
        elif test_gender_ratings[user] == 'male':
            user_gender_rating = male_mean
        elif test_gender_ratings[user] == 'unknown':
            user_gender_rating = unknown_mean
        else:
            user_gender_rating = both_mean
        try:
            user_review_rating = parse_avg[user]
        except KeyError:
            user_review_rating = mean_stars
        try:
            review_count_reviews = len(review_user_stars[user])
        except KeyError:
            review_count_reviews = 1
        rating = (user_gender_rating + user_review_rating*np.log(review_count_reviews))/(np.log(review_count_reviews)
                                                                                         + 1)
        user_ratings[user] = rating

    for user in all_groups:
        if test_gender_ratings[user] == 'female':
            user_gender_rating = female_mean
        elif test_gender_ratings[user] == 'male':
            user_gender_rating = male_mean
        elif test_gender_ratings[user] == 'unknown':
            user_gender_rating = unknown_mean
        else:
            user_gender_rating = both_mean
        try:
            user_review_rating = parse_avg[user]
        except KeyError:
            user_review_rating = mean_stars
        try:
            user_stars = id_stars_dict[user]
        except KeyError:
            user_stars = mean_stars
        user_stars_review = review_stars_average[user]
        review_count = id_reviews[user]
        review_count_reviews = len(review_user_stars[user])
        fuc_rating = fuc_rating_dict[user]
        fuc_count = total_fuc_ratings[user]
        rating = ((np.log(review_count_reviews)*user_review_rating + user_stars*np.log(review_count)
                  + user_stars_review*np.log(review_count_reviews) + user_gender_rating
                  + fuc_rating*np.log(fuc_count))/(2*np.log(review_count_reviews)
                                                   + np.log(review_count)
                                                   + np.log(fuc_count) + 1))
        user_ratings[user] = rating

    #business stuff, fill this
    for business in training_review_businesses:
        rating = mean_stars
        business_ratings[business] = rating
    for business in training_businesses:
        '''business_stars = id_stars_dict[business]
        review_count = id_reviews[user]
        rating = (np.log(review_count)*business_stars + user_gender_rating)/(np.log(review_count) + 1)
        business_ratings[business] = rating'''
        rating = expected_business_rating[business]
        business_ratings[business] = rating
    '''for business in test_businesses:
        user_gender_rating = gender_ratings[user]
        rating = user_gender_rating
        user_ratings[user] = rating
    for user in parse_review_businesses:
        user_review_rating = parse_ratings[user]
        user_stars = stars_average[user]
        review_count = len(review_user_stars[user])
        rating = user_review_rating
    for user in (test_users and training_users):
        user_gender_rating = gender_ratings[user]
        user_stars = id_stars_dict[user]
        review_count = id_reviews[user]
        #rating =
    for user in (parse_review_users and training_users):
        user_review_rating = parse_ratings[user]
        #rating =
    for user in (parse_review_users and test_users):
        review_count = test_count
        user_gender_rating = gender_ratings[user]
        #rating =
    for user in all_groups:
        user_gender_rating = gender_ratings[user]
        user_review_rating = parse_ratings[user]
        fuc_rating = fuc_rating_dict[user]
        user_stars = id_stars_dict[user]
        review_count = id_reviews[user]
        #rating ='''
    with open('yelp_test_set_review.json') as file:
        final_data = [json.loads(entry) for entry in file]
        ratings = []
        for i in range(len(final_data)):
            try:
                rating = (user_ratings[final_data[i]['user_id']] + business_ratings[final_data[i]['business_id']])/2
            except KeyError:
                rating = (user_ratings[final_data[i]['user_id']])
            ratings.append({'RecommendationId': i+1, 'Stars': rating})
    keys = ['RecommendationId', 'Stars']
    f = open('complex.csv', 'w')
    dict_writer = csv.DictWriter(f, keys)
    dict_writer.writer.writerow(keys)
    dict_writer.writerows(ratings)

if __name__ == '__main__':
    main()