__author__ = 'slyfocks'
#miscellaneous user analysis
import numpy as np
import json
import review_parse as rp

with open('yelp_test_set_user.json') as file:
    test_user_data = [json.loads(user) for user in file]

with open('yelp_training_set_user.json') as file:
    training_user_data = [json.loads(user) for user in file]

with open('yelp_test_set_review.json') as file:
    test_review_users = [json.loads(entry)['user_id'] for entry in file]


def training_users():
    return [entry['user_id'] for entry in training_user_data]


def test_users():
    return [entry['user_id'] for entry in test_user_data]


#finds the intersection of training_users and test_review_users: len = 6216
def review_training_users():
    return [id for id in training_users() if id in test_review_users]


#finds the intersection of test_users and test_review_users: len = 5105
def review_test_users():
    return [user_id for user_id in test_users() if user_id in test_review_users]


#finds the intersection of training_review users and test_review users: len = 6611
def review_training_test_users():
    training_review_id_list = rp.training_review_ids()
    return [user_id for user_id in training_review_id_list if user_id in test_review_users]

