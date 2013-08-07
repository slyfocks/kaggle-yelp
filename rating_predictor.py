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

#right now the code is agnostic to business data. coming shortly...
with open('yelp_test_set_user.json') as file:
    data = [json.loads(line) for line in file]
#convert names to all lowercase for alphabetizing purposes
id_names = {entry['user_id']: entry['name'].strip().lower() for entry in data}

with open('yelp_test_set_review.json') as file:
    review_data = [json.loads(line) for line in file]
user_ids = [entry['user_id'] for entry in review_data]
business_ids = [entry['business_id'] for entry in review_data]

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
    id_grades = rp.id_grades()
    id_star = gender.id_stars()
    grades_keys = id_grades.keys()
    star_keys = id_star.keys()
    return [(id_grades[id], id_star[id]) for id in user_ids if id in (grades_keys and star_keys)]


def stars_grade_lists():
    stars_grades_tuples = stars_grades()
    return [list(t) for t in zip(*stars_grades_tuples)]


def reviews_grades():
    id_grades = rp.id_grades()
    id_review = gender.id_reviews()
    grades_keys = id_grades.keys()
    review_keys = id_review.keys()
    return [(id_grades[id], id_review[id]) for id in user_ids if id in (grades_keys and review_keys)]


def reviews_grade_lists():
    reviews_grades_tuples = reviews_grades()
    return [list(t) for t in zip(*reviews_grades_tuples)]


#star rating and grade_level for a particular review
def grade_star_lists():
    grades = rp.grade_stars().keys()
    stars = list(rp.grade_stars().values())
    return [grades, stars]


def test_id_reviews():
    return {member['user_id']: member['review_count'] for member in data}


def mess():
    mean_stars = fuc.mean_user_stars()
    name_list = names(user_ids)
    business_ratings = banal.predicted_business_rating()
    gender_ratings = gender_means_recommendation(genders(name_list))
    business_ids = [entry['business_id'] for entry in review_data]
    business_review_dict = banal.review_counts_dict()
    fuc_dict = fuc.predicted_rating()
    user_review_average = gender.avg_user_reviews()
    bus_review_average = banal.avg_review_counts()
    avg_fuc = fuc.mean_fuc()
    id_stars_dict = gender.id_stars()
    fuc_count = fuc.total_fuc()
    review_count_dict = test_id_reviews()
    training_review_count_dict = gender.id_reviews()
    for i in range(len(review_data)):
        try:
            business_rating = business_ratings[business_ids[i]]
            business_review_counts = business_review_dict[business_ids[i]]
            fuc_rating = fuc_dict[user_ids[i]]
            user_fuc = fuc_count[user_ids[i]]
            #user_stars = id_stars_dict[user_ids[i]]
            #user_review_count = review_count_dict[user_ids[i]]
        except KeyError:
            business_rating = gender_ratings[i]['Stars']
            business_review_counts = 1
            user_fuc = avg_fuc
            fuc_rating = mean_stars
            #user_stars = mean_stars
            '''try:
                user_review_count = training_review_count_dict[user_ids[i]]
            except KeyError:
                print('goop')
                user_review_count = user_review_average'''
        gender_rating = gender_ratings[i]['Stars']
        '''rating_numerator = ((bus_review_average + user_fuc)*fuc_rating
                         + (business_review_counts + avg_fuc)*business_rating
                         + (gender_rating - mean_stars))
        rating_denominator = (business_review_counts + avg_fuc) + (bus_review_average + user_fuc)'''
        #overall_rating = rating_numerator/rating_denominator
        overall_rating = business_rating*gender_rating/mean_stars
        gender_ratings[i]['Stars'] = overall_rating
    keys = ['RecommendationId', 'Stars']
    f = open('businesspeoplemod.csv', 'w')
    dict_writer = csv.DictWriter(f, keys)
    dict_writer.writer.writerow(keys)
    dict_writer.writerows(gender_ratings)


def main():
    #user_ids is test set_review_ids
    name_list = names(user_ids)
    gender_ratings = gender_means_recommendation(genders(name_list))
    test_count = data['review_count']
    mean_stars = fuc.mean_user_stars()[user]

    #for users in training user set
    id_stars_dict = gender.id_stars()
    id_reviews = gender.id_reviews()

    #user_id stuff
    test_users = ua.review_test_users()
    training_users = ua.review_training_users()
    all_groups = ua.all_group_users()

    #these variables will be for users who have writing samples available
    parse_review_users = ua.review_training_test_users()
    parse_ratings = ua.user_review_parse_rating()
    review_user_stars = rp.id_stars()
    stars_average = rp.id_stars_avg()

    #business stuff
    test_businesses = banal.review_test_businesses()
    training_businesses = banal.review_test_businesses()
    test_categories = [banal.categories(entry) for entry in test_businesses]
    training_categories = [banal.categories(entry) for entry in training_businesses]

    #funny_useful_cool stuff
    fuc_rating_dict = fuc.predicted_rating()
    #where final ratings go for users
    user_ratings = {}

    #where final ratings go for businesses
    business_ratings = {}


    #in case all of these loops don't contain certain users, initialize all users to the mean
    for user in user_ids:
        user_ratings[user] = mean_stars

    for user in training_users:
        user_gender_rating = gender_ratings[user]
        user_stars = id_stars_dict[user]
        review_count = id_reviews[user]
        rating = (np.log(review_count)*user_stars + user_gender_rating)/(np.log(review_count) + 1)
        user_ratings[user] = rating
    for user in test_users:
        user_gender_rating = gender_ratings[user]
        rating = user_gender_rating
        user_ratings[user] = rating
    for user in parse_review_users:
        user_review_rating = parse_ratings[user]
        user_stars = stars_average[user]
        review_count = len(review_user_stars[user])
        rating = user_review_rating
    for user in (test_users and training_users):
        user_gender_rating = gender_ratings[user]
        user_stars = id_stars_dict[user]
        review_count = id_reviews[user]
        rating = (np.log(review_count)*user_stars + user_gender_rating)/(np.log(review_count) + 1)
        user_ratings[user] = rating
    for user in (parse_review_users and training_users):
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
        #rating =

    #business stuff, fill this
    for business in training_businesses:
        business_gender_rating = gender_ratings[business]
        business_stars = id_stars_dict[business]
        review_count = id_reviews[user]
        rating = (np.log(review_count)*user_stars + user_gender_rating)/(np.log(review_count) + 1)
        business_ratings[business] = rating
    for user in test_users:
        user_gender_rating = gender_ratings[user]
        rating = user_gender_rating
        user_ratings[user] = rating
    for user in parse_review_users:
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
        #rating =
    keys = ['RecommendationId', 'Stars']
    f = open('businesspeoplemod.csv', 'w')
    dict_writer = csv.DictWriter(f, keys)
    dict_writer.writer.writerow(keys)
    dict_writer.writerows(gender_ratings)

if __name__ == '__main__':
    main()