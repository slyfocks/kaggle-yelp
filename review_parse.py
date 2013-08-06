__author__ = 'slyfocks'
import nltk
import json
import numpy as np
import csv
import gender
import matplotlib.pyplot as plt
from grade_level import grade_level
from mpl_toolkits.mplot3d import Axes3D


#this takes a LONG time to run...only run this once to make the csv file, then read off of that.
def write_id_grades():
    with open('yelp_training_set_review.json') as file:
        outfile = csv.writer(open('grade_id_pairs.csv', 'w'))
        for review in file:
            outfile.writerow([json.loads(review)['user_id'],
                              str(grade_level(json.loads(review)['text'])),
                              json.loads(review)['stars'],
                              json.loads(review)['business_id']])


#returns a dictionary of user_ids with paired with a list of the grade levels of their reviews
def id_grades():
    with open('grade_id_pairs.csv') as file:
        contents = csv.reader(file, delimiter=',')
        id_grade_dict = {}
        #initalize a dict entry to empty list for each user since some users have multiple reviews
        for entry in contents:
            #if it has values already, append another. otherwise, create the entry
            try:
                id_grade_dict[entry[0]].append(float(entry[1]))
            except KeyError:
                id_grade_dict[entry[0]] = [float(entry[1])]
    return id_grade_dict


#makes list of user_ids in training review set. no duplicates!
def training_review_ids():
    id_grade_dict = id_grades()
    return [user_id for user_id in id_grade_dict.keys()]


#returns dict of user_ids and their average review grade level
def id_grade_avg():
    grade_dict = id_grades()
    avg_grade_dict = {}
    for user_id in list(grade_dict.keys()):
        grade_list = grade_dict[user_id]
        avg_grade_dict[user_id] = sum(grade_list)/len(grade_list)
    return avg_grade_dict


def id_stars():
    with open('grade_id_pairs.csv') as file:
        contents = csv.reader(file, delimiter=',')
        id_star_dict = {}
        for entry in contents:
            #if it has values already, append another. otherwise, create the entry
            try:
                id_star_dict[entry[0]].append(float(entry[2]))
            except KeyError:
                id_star_dict[entry[0]] = [float(entry[2])]
    return id_star_dict


def id_stars_avg():
    stars_dict = id_stars()
    avg_stars_dict = {}
    for user_id in list(stars_dict.keys()):
        stars_list = stars_dict[user_id]
        avg_stars_dict[user_id] = sum(stars_list)/len(stars_list)
    return avg_stars_dict


def grades_stars():
    ids = training_review_ids()
    id_star_dict = id_stars()
    id_grade_dict = id_grades()
    stars = []
    grades = []
    for user_id in ids:
        for star in id_star_dict[user_id]:
            stars.append(star)
        for grade in id_grade_dict[user_id]:
            grades.append(grade)
    return [grades, stars]


def sort_grades():
    grade_star_lists = grades_stars()
    return list(np.sort(grade_star_lists[0]))


#partition the grade domain into areas so properties of each range can be analyzed
def grade_range_star_dict():
    grades_stars_list = grades_stars()
    grades = grades_stars_list[0]
    stars = grades_stars_list[1]
    grade_star_dict = {'-20': [], '-4': [], '0': [], '3': [], '5': [], '8': [], '13': [], '23': []}
    for i in range(len(grades_stars_list[0])):
        if -20 < grades[i] < -4:
            grade_star_dict['-20'].append(stars[i])
        elif -4 < grades[i] < 0:
            grade_star_dict['-4'].append(stars[i])
        elif 0 < grades[i] < 3:
            grade_star_dict['0'].append(stars[i])
        elif 3 < grades[i] < 5:
            grade_star_dict['3'].append(stars[i])
        elif 5 < grades[i] < 8:
            grade_star_dict['5'].append(stars[i])
        elif 8 < grades[i] < 13:
            grade_star_dict['8'].append(stars[i])
        elif 13 < grades[i] < 23:
            grade_star_dict['13'].append(stars[i])
        else:
            grade_star_dict['23'].append(stars[i])
    return grade_star_dict


def grades_stars_avg():
    ids = training_review_ids()
    id_star_dict = id_stars_avg()
    id_grade_dict = id_grade_avg()
    return [(id_grade_dict[id], id_star_dict[id]) for id in ids]


def grades_stars_restricted(num):
    ids = training_review_ids()
    id_star_dict = id_stars_avg()
    id_grade_dict = id_grade_avg()
    return [(id_grade_dict[id], id_star_dict[id]) for id in ids if id_grade_dict[id] < num]


#turns tuples into two lists by zipping
def grade_star_lists():
    grade_star_tuples = grades_stars_avg()
    grades, stars = zip(*grade_star_tuples)
    return [list(grades), list(stars)]


def gender_grades():
    gender_grades = {'female': [], 'male': [], 'unknown': [], 'both': []}
    ids = id_grades().keys()
    id_gender = gender.id_gender()
    id_grade_dict = id_grades()
    for user_id in ids:
        try:
            #loop through the grade values of the reviews for each user_id
            for i in range(len(id_grade_dict[user_id])):
                gender_grades[id_gender[user_id]].append(float(id_grade_dict[user_id][i]))
        except KeyError:
            continue
    return gender_grades


def gender_average_grade():
    grades = gender_grades()

    female_average = sum(grades['female'])/len(grades['female'])
    female_std = np.sqrt(np.var(grades['female'])/(len(grades['female']) - 1))

    male_average = sum(grades['male'])/len(grades['male'])
    male_std = np.sqrt(np.var(grades['male'])/(len(grades['male']) - 1))

    both_average = sum(grades['both'])/len(grades['both'])
    both_std = np.sqrt(np.var(grades['both'])/(len(grades['both']) - 1))

    unknown_average = sum(grades['unknown'])/len(grades['unknown'])
    unknown_std = np.sqrt(np.var(grades['unknown'])/(len(grades['unknown']) - 1))
    return [(female_average, female_std), (male_average, male_std),
            (both_average, both_std), (unknown_average, unknown_std)]


def grade_star_regression():
    x = np.array(grades_stars()[0][:40000])
    y = np.array(grades_stars()[1][:40000])
    A = np.vstack([x, np.ones(len(x))]).T
    m, c = np.linalg.lstsq(A, y)[0]
    plt.scatter(x,y)
    return m,c
print(grade_star_regression())