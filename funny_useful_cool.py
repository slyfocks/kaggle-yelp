__author__ = 'slyfocks'
import json
import matplotlib.pyplot as plt
import numpy as np

with open('yelp_training_set_user.json') as file:
    user_data = [json.loads(user) for user in file]


def user_ids():
    return [user['user_id'] for user in user_data]


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


def total_fuc():
    fuc_dict = fuc_scores()
    ids = user_ids()
    return {id: sum(fuc_dict[id]) for id in ids}


def mean_fuc():
    overall_fuc = 0
    for user in user_data:
        funny_count = user['votes']['funny']
        useful_count = user['votes']['useful']
        cool_count = user['votes']['cool']
        fuc_sum = funny_count + useful_count + cool_count
        overall_fuc += fuc_sum
    return overall_fuc/len(user_data)


def user_stars():
    return {user['user_id']: user['average_stars'] for user in user_data}


def mean_user_stars():
    user_stars_list = user_stars().values()
    return sum(user_stars_list)/len(user_stars_list)


def fuc_scores_stars():
    fuc_score_dict = fuc_scores()
    user_star_dict = user_stars()
    ids = user_ids()
    return [(fuc_score_dict[id], user_star_dict[id]) for id in ids]


#enter funny, useful, or cool as string
def fuc_xor_scores_stars(fuc):
    score_stars_list = fuc_scores_stars()
    if fuc == 'funny':
        return [(entry[0][0], entry[1]) for entry in score_stars_list]
    elif fuc == 'useful':
        return [(entry[0][1], entry[1]) for entry in score_stars_list]
    else:
        return [(entry[0][2], entry[1]) for entry in score_stars_list]


#e.g. score_stars_lists('cool')[0] is the list of cool scores
def score_stars_lists(fuc):
    tuple_list = fuc_xor_scores_stars(fuc)
    return [list(t) for t in zip(*tuple_list)]


def fuc_regression(fuc):
    x = np.array(score_stars_lists(fuc)[0])
    y = np.array(score_stars_lists(fuc)[1])
    A = np.vstack([x, np.ones(len(x))]).T
    m, c = np.linalg.lstsq(A, y)[0]
    return m, c


#if a user has not rated many things, adjust their average stars through Bayesian smoothing
#this should now be used for each user's average stars!!
def predicted_rating():
    ids = user_ids()
    fuc_dict = fuc_scores()
    rating_dict = {}
    mf, cf = fuc_regression('funny')
    mu, cu = fuc_regression('useful')
    mc, cc = fuc_regression('cool')
    for id in ids:
        #prediction based on funny slope and intercept and funny score
        funny_stars = fuc_dict[id][0]*mf + cf
        #prediction based on useful slope and intercept and useful score
        useful_stars = fuc_dict[id][1]*mu + cu
        #prediction based on cool slope and intercept and cool score
        cool_stars = fuc_dict[id][2]*mc + cc
        predicted_stars = (funny_stars + useful_stars + cool_stars)/3
        rating_dict[id] = predicted_stars
    return rating_dict
