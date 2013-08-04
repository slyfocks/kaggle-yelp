__author__ = 'slyfocks'
import nltk
import json
import csv
from grade_level import grade_level


def write_id_grades():
    with open('yelp_training_set_review.json') as file:
        outfile = csv.writer(open('grade_id_pairs.csv', 'w'))
        for review in file:
            outfile.writerow([json.loads(review)['user_id'], str(grade_level(json.loads(review)['text']))])
