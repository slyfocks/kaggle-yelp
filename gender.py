__author__ = 'slyfocks'
import json
import numpy as np
from gender_guesser.name_gender import NameGender
from pandas import *
import matplotlib.pyplot as plt
import pylab
with open('yelp_training_set_user.json') as file:
    data = [json.loads(line) for line in file]
stars_reviews = [{'average_stars': member['average_stars'], 'review_count': member['review_count']} for member in data]
#str.lower is used so lowercase names appear in the right order alongside uppercase names
sorted_stars_reviews = sorted(stars_reviews, key=lambda x: x.keys())
names = [member['name'] for member in data]
stars = [member['average_stars'] for member in stars_reviews]
reviews = [member['review_count'] for member in stars_reviews]


def id_stars():
    return {member['user_id']: member['average_stars'] for member in data}


def get_decision(guesser, name):
    m, f = guesser.get_gender_scores(name)
    if m > 0.8:
        return "male"
    elif f > 0.8:
        return "female"
    elif m > 0 and f > 0:
        return "both"
    else:
        return "unknown"


def name_gender(names):
    primary_guesser = NameGender("gender_guesser/us_census_1990_males", "gender_guesser/us_census_1990_females")
    secondary_guesser = NameGender("gender_guesser/popular_1960_2010_males", "gender_guesser/popular_1960_2010_females")
    name_genders = []
    for name in names:
        gender = get_decision(primary_guesser, name)
        if gender in ["male", "female", "both"]:
            name_genders.append(gender)
        else:
            gender = get_decision(secondary_guesser, name)
            if gender in ["male", "female", "both"]:
                name_genders.append(gender)
            else:
                name_genders.append('unknown')
    return name_genders


#returns user_id:gender pairs in dict form
def id_gender():
    primary_guesser = NameGender("gender_guesser/us_census_1990_males", "gender_guesser/us_census_1990_females")
    secondary_guesser = NameGender("gender_guesser/popular_1960_2010_males", "gender_guesser/popular_1960_2010_females")
    id_gender = {}
    i = 0
    j = 0
    for member in data:
        name = member['name'].strip().lower()
        gender = get_decision(primary_guesser, name)
        if gender in ["male", "female", "both"]:
            id_gender[member['user_id']] = gender
        else:
            gender = get_decision(secondary_guesser, name)
            if gender in ["male", "female", "both"]:
                id_gender[member['user_id']] = gender
            else:
                id_gender[member['user_id']] = 'unknown'
    return id_gender


def data_genders(data):
    # Give precedence to us_census data.
    primary_guesser = NameGender("gender_guesser/us_census_1990_males", "gender_guesser/us_census_1990_females")
    secondary_guesser = NameGender("gender_guesser/popular_1960_2010_males", "gender_guesser/popular_1960_2010_females")
    dict_genders = []
    for entry in data:
        name = entry['name'].strip().lower()
        gender = get_decision(primary_guesser, name)
        if gender in ["male", "female", "both"]:
            dict_genders.append({(entry['review_count'], entry['average_stars']): gender})
        else:
            gender = get_decision(secondary_guesser, name)
            if gender in ["male", "female", "both"]:
                dict_genders.append({(entry['review_count'], entry['average_stars']): gender})
            else:
                dict_genders.append({(entry['review_count'], entry['average_stars']): 'unknown'})
    return dict_genders


def gender_tuples(data):
    return [[list(entry.keys())[0] for entry in data_genders(data) if list(entry.values())[0] == 'female'],
            [list(entry.keys())[0] for entry in data_genders(data) if list(entry.values())[0] == 'male'],
            [list(entry.keys())[0] for entry in data_genders(data) if list(entry.values())[0] == 'unknown'],
            [list(entry.keys())[0] for entry in data_genders(data) if list(entry.values())[0] == 'both']]


def female_data(data):
    female_zip = list(zip(*gender_tuples(data)[0]))
    female_reviews = list(female_zip[0])
    female_stars = list(female_zip[1])
    return {'review_count': female_reviews, 'average_stars': female_stars}


def male_data(data):
    male_zip = list(zip(*gender_tuples(data)[1]))
    male_reviews = list(male_zip[0])
    male_stars = list(male_zip[1])
    return {'review_count': male_reviews, 'average_stars': male_stars}


def unknown_data(data):
    unknown_zip = list(zip(*gender_tuples(data)[2]))
    unknown_reviews = list(unknown_zip[0])
    unknown_stars = list(unknown_zip[1])
    return {'review_count': unknown_reviews, 'average_stars': unknown_stars}


def both_data(data):
    both_zip = list(zip(*gender_tuples(data)[3]))
    both_reviews = list(both_zip[0])
    both_stars = list(both_zip[1])
    return {'review_count': both_reviews, 'average_stars': both_stars}


def statistics(data):
    mean_female_x = np.mean(female_data(data)['review_count'])
    #sample variance for mean of sample
    std_female_x = np.std(female_data(data)['review_count'])/(len(female_data(data)['review_count'])-1)

    mean_female_y = np.mean(female_data(data)['average_stars'])
    std_female_y = np.std(female_data(data)['average_stars'])/(len(female_data(data)['average_stars'])-1)

    mean_male_x = np.mean(male_data(data)['review_count'])
    std_male_x = np.std(male_data(data)['review_count'])/(len(male_data(data)['review_count'])-1)

    mean_male_y = np.mean(male_data(data)['average_stars'])
    std_male_y = np.std(male_data(data)['average_stars'])/(len(male_data(data)['average_stars'])-1)

    mean_unknown_x = np.mean(unknown_data(data)['review_count'])
    std_unknown_x = np.std(unknown_data(data)['review_count'])/(len(unknown_data(data)['review_count'])-1)

    mean_unknown_y = np.mean(unknown_data(data)['average_stars'])
    std_unknown_y = np.std(unknown_data(data)['average_stars'])/(len(unknown_data(data)['average_stars'])-1)

    mean_both_x = np.mean(both_data(data)['review_count'])
    std_both_x = np.std(both_data(data)['review_count'])/(len(both_data(data)['review_count'])-1)

    mean_both_y = np.mean(both_data(data)['average_stars'])
    std_both_y = np.std(both_data(data)['average_stars'])/(len(both_data(data)['average_stars'])-1)
    return {'female': [(mean_female_x, std_female_x), (mean_female_y, std_female_y)],
            'male': [(mean_male_x, std_male_x), (mean_male_y, std_male_y)],
            'unknown': [(mean_unknown_x, std_unknown_x), (mean_unknown_y, std_unknown_y)],
            'both': [(mean_both_x, std_both_x), (mean_both_y, std_both_y)]}


def training_mean(gender):
    #returns mean stars for each gender
    return statistics(data)[gender][1][0]
