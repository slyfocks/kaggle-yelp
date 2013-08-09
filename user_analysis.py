__author__ = 'slyfocks'
#miscellaneous user analysis
import numpy as np
import json
import review_parse as rp
import csv

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


def training_review_ids():
    with open('grade_id_pairs.csv') as file:
        contents = csv.reader(file, delimiter=',')
        users = [review[0] for review in contents]
    return users


#finds the intersection of training_review users and test_review users: len = 6611
def review_training_test_users():
    training_review_id_list = training_review_ids()
    return [user_id for user_id in set(training_review_id_list).intersection(test_review_users)]


def all_group_users():
    test_user_list = test_users()
    training_user_list = training_users()
    training_review_id_list = training_review_ids()
    return [user_id for user_id in training_review_id_list
            if user_id in set(training_user_list).intersection(test_user_list)]


#takes in user_ids, outputs dict of user_ids and predicted mean
#this function is messing things up, feel free to ignore it
def user_review_parse_rating():
    user_grades = rp.id_grade_avg()
    partitions = rp.partitions()
    #this is a dict, give it a partition number
    partition_mean_stds = rp.partition_mean_std()
    user_rating_dict = {}
    for user_id in review_training_test_users():
        grade = user_grades[user_id]
        for i in range(len(partitions)):
            if float(partitions[i]) > 115:
                if grade > 9.25296536797:
                    user_rating_dict[user_id] = partition_mean_stds[partitions[i]][0]
                break
            if grade < float(partitions[i]):
                user_rating_dict[user_id] = partition_mean_stds[partitions[i]][0]
                break
    return user_rating_dict