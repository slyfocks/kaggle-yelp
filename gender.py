__author__ = 'slyfocks'
import json
import numpy as np
from gender_guesser.name_gender import NameGender
from pandas import *
import matplotlib.pyplot as plt
with open('yelp_training_set_user.json') as file:
    data = [json.loads(line) for line in file]
stars_reviews = [{'average_stars': member['average_stars'], 'review_count': member['review_count']} for member in data]
#str.lower is used so lowercase names appear in the right order alongside uppercase names
sorted_stars_reviews = sorted(stars_reviews, key=lambda x: x.keys())
names = [member['name'] for member in data]
stars = [member['average_stars'] for member in stars_reviews]
reviews = [member['review_count'] for member in stars_reviews]


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


def name_genders(data):
    # Give precedence to us_census data.
    primary_guesser = NameGender("gender_guesser/us_census_1990_males", "gender_guesser/us_census_1990_females")
    secondary_guesser = NameGender("gender_guesser/popular_1960_2010_males", "gender_guesser/popular_1960_2010_females")
    web_guesser = None
    name_genders = []
    for entry in data:
        name = entry['name'].strip().lower()
        gender = get_decision(primary_guesser, name)
        if gender in ["male", "female", "both"]:
            name_genders.append({(entry['review_count'], entry['average_stars']): gender})
        else:
            gender = get_decision(secondary_guesser, name)
            if gender in ["male", "female", "both"]:
                name_genders.append({(entry['review_count'], entry['average_stars']): gender})
            else:
                name_genders.append({(entry['review_count'], entry['average_stars']): 'unknown'})
    return name_genders


def gender_tuples(data):
    return [[list(entry.keys())[0] for entry in name_genders(data) if list(entry.values())[0] == 'female'],
            [list(entry.keys())[0] for entry in name_genders(data) if list(entry.values())[0] == 'male'],
            [list(entry.keys())[0] for entry in name_genders(data) if list(entry.values())[0] == 'unknown']]


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


def statistics(data):
    mean_female_x = np.mean(female_data(data)['review_count'])
    #sample variance for mean of sample
    var_female_x = np.var(female_data(data)['review_count'])/(len(female_data(data)['review_count'])-1)
    mean_female_y = np.mean(female_data(data)['review_count'])
    var_female_y = np.var(female_data(data)['average_stars'])/(len(female_data(data)['average_stars'])-1)
    return {'female': [(mean_female_x, var_female_x), (mean_female_y, var_female_y)],
            'male': [(mean_male_x, var_male_x), (mean_male_y, var_male_y)]}


mean_male_x = np.mean(male_x)
var_male_x = np.var(male_x)/(len(male_x)-1)

mean_male_y = np.mean(male_y)
var_male_y = np.var(male_y)/(len(male_y)-1)

mean_unknown_x = np.mean(unknown_x)
var_unknown_x = np.var(unknown_x)/(len(unknown_x)-1)

mean_unknown_y = np.mean(unknown_y)
var_unknown_y = np.var(unknown_y)/(len(unknown_y)-1)

print(mean_male_x)
print(mean_male_y)
print(mean_female_x)
print(mean_female_y)
print(mean_unknown_x)
print(mean_unknown_y)
print(np.sqrt(var_male_x))
print(np.sqrt(var_male_y))
print(np.sqrt(var_female_x))
print(np.sqrt(var_female_y))
print(np.sqrt(var_unknown_x))
print(np.sqrt(var_unknown_y))
print(np.mean(stars))
print(np.mean(reviews))
print(np.std(stars))
print(np.std(reviews))
'''plt.scatter(female_y, female_x)
plt.show()'''