__author__ = 'slyfocks'
import json
import numpy as np
import gender
#right now the code is agnostic to business data. coming shortly...
with open('yelp_test_set_user.json') as file:
    data = [json.loads(line) for line in file]
#convert names to all lowercase for alphabetizing purposes
id_names = {entry['user_id']: entry['name'].strip().lower() for entry in data}

with open('yelp_test_set_review.json') as file:
    review_data = [json.loads(line) for line in file]
user_ids = [entry['user_id'] for entry in review_data]


def genders(user_ids):
    genders = []
    for user_id in user_ids:
        try:
            genders.append(gender.name_gender(id_names[user_id]))
        except KeyError:
            genders.append('unknown')
    return genders


def gender_means(gender_list):
    return [gender.training_mean(gender) for gender in gender_list]


def main():
    ratings_array = np.asarray(gender_means(genders(user_ids)))
    np.savetxt('ratings.csv', ratings_array, delimeter=',')

if __name__ == '__main__':
    main()