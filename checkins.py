__author__ = 'slyfocks'
import json
import numpy as np
import business_analysis as banal
with open('yelp_test_set_checkin.json') as file:
    test_checkin_data = [json.loads(entry) for entry in file]

with open('yelp_training_set_checkin.json') as file:
    training_checkin_data = [json.loads(entry) for entry in file]

with open('yelp_test_set_review.json') as file:
    review_businesses = [json.loads(entry)['business_id'] for entry in file]


def test_checkin_business_ids():
    return [entry['business_id'] for entry in test_checkin_data]


def training_checkin_business_ids():
    return [entry['business_id'] for entry in training_checkin_data]


#displays the business_ids that are in the training set and final review set
def review_training_businesses():
    return [rating for rating in banal.test_businesses() if rating in review_businesses]
