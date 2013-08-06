__author__ = 'slyfocks'
import json
import numpy as np
import funny_useful_cool as fuc

with open('yelp_training_set_business.json') as file:
    training_business_data = [json.loads(line) for line in file]

with open('yelp_test_set_business.json') as file:
    test_business_data = [json.loads(line) for line in file]

with open('yelp_test_set_review.json') as file:
    review_businesses = [json.loads(entry)['business_id'] for entry in file]


def training_businesses():
    return [entry['business_id'] for entry in training_business_data]


def test_businesses():
    return [entry['business_id'] for entry in test_business_data]


def training_review_counts():
    return [entry['review_count'] for entry in training_business_data]


def test_review_counts():
    return [entry['review_count'] for entry in test_business_data]


def review_counts_dict():
    return {entry['business_id']: entry['review_count'] for entry in test_business_data}


#output of this function is 20.19
def avg_training_review_counts():
    return sum(training_review_counts())/len(training_review_counts())


#output of this function is 9.19. WAY DIFFERENT THAN TRAINING SET
def avg_test_review_counts():
    return sum(test_review_counts())/len(test_review_counts())


#displays the business_ids that are in the training set and final review set: len = 4380
def review_training_businesses():
    return [id for id in training_businesses() if id in review_businesses]


#displays the business_ids that are in the training set and final review set: len = 1205
def review_test_businesses():
    return [id for id in test_businesses() if id in review_businesses]


def categories(entry):
    return entry['categories']


def category_set(business_data):
    #puts the categories in each of the lists into one master list without repetition
    return set([category for entry in business_data for category in categories(entry)])


def category_rating(category_set):
    category_dict = {category: [] for category in category_set}
    for entry in training_business_data:
        stars = entry['stars']
        review_count = entry['review_count']
        for category in entry['categories']:
            category_dict[category].append([stars, review_count])
    return category_dict


def average_category_rating():
    star_rating_dict = category_rating(category_set(training_business_data))
    average_rating_dict = {}
    for category in category_set(training_business_data):
        total_rating = 0
        total_reviews = 0
        for rating_pairs in star_rating_dict[category]:
            total_rating += rating_pairs[0] * rating_pairs[1]
            total_reviews += rating_pairs[1]
        average_rating_dict[category] = total_rating / total_reviews
    return average_rating_dict


def predicted_business_rating():
    id_rating_dict = {}
    mean = fuc.mean_user_stars()
    category_ratings = average_category_rating()
    for entry in training_business_data:
        actual_rating = entry['stars']
        sum_expected_rating = sum([category_ratings[category] for category in categories(entry)])
        num_categories = len(categories(entry))
        try:
            expected_rating = sum_expected_rating / num_categories
        except ZeroDivisionError:
            expected_rating = (mean + actual_rating)/2
        review_count = entry['review_count']
        predicted_rating = (actual_rating*review_count + expected_rating)/(review_count + 1)
        id_rating_dict[entry['business_id']] = predicted_rating
    for entry in test_business_data:
        try:
            predicted_rating = id_rating_dict[entry['business_id']]
        except KeyError:
            sum_expected_rating = sum([category_ratings[category] for category in categories(entry)
                                       if category in category_set(training_business_data)])
            num_categories = len(categories(entry))
            try:
                expected_rating = sum_expected_rating / num_categories
            except ZeroDivisionError:
                expected_rating = mean
            predicted_rating = expected_rating
            id_rating_dict[entry['business_id']] = predicted_rating
    return id_rating_dict



