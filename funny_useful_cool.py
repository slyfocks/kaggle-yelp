__author__ = 'slyfocks'
import json
import matplotlib.pyplot as plt

with open('yelp_training_set_user.json') as file:
    user_data = [json.loads(user) for user in file]


def fuc_scores():
    fuc_dict = {}
    for user in user_data:
        funny_count = user['votes']['funny']
        useful_count = user['votes']['useful']
        cool_count = user['votes']['cool']
        funny_score = funny_count/(funny_count + useful_count + cool_count + 1)
        useful_score = useful_count/(funny_count + useful_count + cool_count + 1)
        cool_score = cool_count/(funny_count + useful_count + cool_count + 1)
        fuc_dict[user['user_id']] = [funny_score, useful_score, cool_score]
    return fuc_dict