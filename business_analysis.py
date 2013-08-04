__author__ = 'slyfocks'
import json
import numpy as np

with open('yelp_training_set_business.json') as file:
    business_data = [json.loads(line) for line in file]
print(business_data)


def categories(entry):
    return entry['categories']


def category_set(business_data):
    #puts the categories in each of the lists into one master list without repetition
    return set([category for entry in business_data for category in categories(entry)])


def category_rating(category_set):
    category_dict = {category: [] for category in category_set}
    for entry in business_data:
        stars = entry['stars']
        review_count = entry['review_count']
        for category in entry['categories']:
            category_dict[category].append([stars, review_count])
    return category_dict


def average_category_rating():
    star_rating_dict = category_rating(category_set(business_data))
    average_rating_dict = {}
    for category in category_set(business_data):
        total_rating = 0
        total_reviews = 0
        for rating_pairs in star_rating_dict[category]:
            total_rating += rating_pairs[0] * rating_pairs[1]
            total_reviews += rating_pairs[1]
        average_rating_dict[category] = total_rating / total_reviews
    return average_rating_dict


def predicted_business_rating():
    id_rating_dict = {}
    category_ratings = average_category_rating()
    for entry in business_data:
        actual_rating = entry['stars']
        sum_expected_rating = sum([category_ratings[category] for category in categories(entry)])
        num_categories = len(categories(entry))
        try:
            expected_rating = sum_expected_rating / num_categories
        except ZeroDivisionError:
            expected_rating = actual_rating
        review_count = entry['review_count']
        predicted_rating = (actual_rating*review_count + expected_rating)/(review_count + 1)
        id_rating_dict[entry['business_id']] = predicted_rating
    return id_rating_dict

print(predicted_business_rating())


