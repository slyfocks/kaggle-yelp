__author__ = 'slyfocks'
import nltk
import json
import numpy as np
import csv
import gender
from grade_level import grade_level


def write_id_grades():
    with open('yelp_training_set_review.json') as file:
        outfile = csv.writer(open('grade_id_pairs.csv', 'w'))
        for review in file:
            outfile.writerow([json.loads(review)['user_id'],
                              str(grade_level(json.loads(review)['text'])),
                              json.loads(review)['stars']])


def id_grades():
    with open('grade_id_pairs.csv') as file:
        contents = csv.reader(file, delimiter=',')
        id_grade_dict = {}
        for entry in contents:
            id_grade_dict[entry[0]] = entry[1]
    return id_grade_dict


def grade_stars():
    with open('grade_id_pairs.csv') as file:
        contents = csv.reader(file, delimiter=',')
        grade_star_dict = {}
        for entry in contents:
            grade_star_dict[entry[1]] = entry[2]
    return grade_star_dict


def gender_grades():
    gender_grades = {'female': [], 'male': [], 'unknown': [], 'both': []}
    ids = id_grades().keys()
    id_gender = gender.id_gender()
    id_grade_dict = id_grades()
    for id in ids:
        try:
            gender_grades[id_gender[id]].append(float(id_grade_dict[id]))
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
