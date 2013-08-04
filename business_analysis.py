__author__ = 'slyfocks'
import json
import numpy as np

with open('yelp_training_set_business.json') as file:
    data = [json.loads(line) for line in file]
print(data)


def categories(entry):
    return entry['categories']


def category_set(data):
    #puts the categories in each of the lists into one master list without repetition
    return set([category for entry in data for category in categories(entry)])


def category_rating
print(category_set(data))