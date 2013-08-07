__author__ = 'slyfocks'
import json
import numpy as np
import funny_useful_cool as fuc
import csv
import review_parse as rp
import matplotlib.pyplot as plt
from collections import OrderedDict

with open('yelp_training_set_business.json') as file:
    training_business_data = [json.loads(line) for line in file]

with open('yelp_test_set_business.json') as file:
    test_business_data = [json.loads(line) for line in file]

with open('yelp_test_set_review.json') as file:
    review_businesses = [json.loads(entry)['business_id'] for entry in file]


def training_businesses():
    return [entry['business_id'] for entry in training_business_data]


def test_businesses():
    return [entry['business_id'] for entry in test_business_data]


def training_review_counts():
    return [entry['review_count'] for entry in training_business_data]


def test_review_counts():
    return [entry['review_count'] for entry in test_business_data]


def review_counts_dict():
    return {entry['business_id']: entry['review_count'] for entry in test_business_data}


#output of this function is 20.19
def avg_training_review_counts():
    return sum(training_review_counts())/len(training_review_counts())


#output of this function is 9.19. WAY DIFFERENT THAN TRAINING SET
def avg_test_review_counts():
    return sum(test_review_counts())/len(test_review_counts())


#displays the business_ids that are in the training set and final review set: len = 4380
def review_training_businesses():
    return [id for id in training_businesses() if id in review_businesses]


#displays the business_ids that are in the test set and final review set: len = 1205
def review_test_businesses():
    return [id for id in test_businesses() if id in review_businesses]


def categories(entry):
    return entry['categories']


def id_categories():
    return {entry['business_id']: entry['categories'] for entry in training_business_data}


def category_set(business_data):
    #puts the categories in each of the lists into one master list without repetition
    return set([category for entry in business_data for category in categories(entry)])


def category_rating(category_set):
    category_dict = {category: [] for category in category_set}
    for entry in training_business_data:
        stars = entry['stars']
        review_count = entry['review_count']
        for category in entry['categories']:
            category_dict[category].append([stars, review_count])
    return category_dict


def average_category_rating():
    star_rating_dict = category_rating(category_set(training_business_data))
    average_rating_dict = {}
    for category in category_set(training_business_data):
        total_rating = 0
        total_reviews = 0
        for rating_pairs in star_rating_dict[category]:
            total_rating += rating_pairs[0] * rating_pairs[1]
            total_reviews += rating_pairs[1]
        average_rating_dict[category] = total_rating / total_reviews
    return average_rating_dict


#returns a dictionary of business_ids with paired with a list of the grade levels of their reviews
def id_grades():
    with open('grade_id_pairs.csv') as file:
        contents = csv.reader(file, delimiter=',')
        id_grade_dict = {}
        #initalize a dict entry to empty list for each user since some users have multiple reviews
        for entry in contents:
            #if it has values already, append another. otherwise, create the entry
            try:
                id_grade_dict[entry[3]].append(float(entry[1]))
            except KeyError:
                id_grade_dict[entry[3]] = [float(entry[1])]
    return id_grade_dict


#returns dict of business_ids and their average review grade level; only for businesses in training_review set
def id_grade_avg():
    grade_dict = id_grades()
    avg_grade_dict = {}
    for business_id in list(grade_dict.keys()):
        grade_list = grade_dict[business_id]
        avg_grade_dict[business_id] = sum(grade_list)/len(grade_list)
    return avg_grade_dict


def id_stars():
    bus_id_star_dict = {}
    with open('grade_id_pairs.csv') as file:
        contents = csv.reader(file, delimiter=',')
        for entry in contents:
            try:
                bus_id_star_dict[entry[3]].append(float(entry[2]))
            except KeyError:
                bus_id_star_dict[entry[3]] = [float(entry[2])]
    return bus_id_star_dict


def id_star_avg():
    star_dict = id_stars()
    avg_star_dict = {}
    for business_id in list(star_dict.keys()):
        star_list = star_dict[business_id]
        avg_star_dict[business_id] = sum(star_list)/len(star_list)
    return avg_star_dict


#build up dict of business_ids and grade and star differences between business and user averages
def stars_grade_diff():
    business_id_star_dict = id_star_avg()
    user_id_grade_dict = rp.id_grade_avg()
    business_id_grade_dict = id_grade_avg()
    value_dict = {}
    with open('grade_id_pairs.csv') as file:
        contents = csv.reader(file, delimiter=',')
        for entry in contents:
            try:
                #entry[2] is star rating for that particular review
                value_dict[entry[3]].append((float(entry[2]) - business_id_star_dict[entry[3]],
                                             user_id_grade_dict[entry[0]] - float(entry[1]),
                                             business_id_grade_dict[entry[3]] - float(entry[1]),
                                             user_id_grade_dict[entry[0]] - business_id_grade_dict[entry[3]]))
            except KeyError:
                value_dict[entry[3]] = [(float(entry[2]) - business_id_star_dict[entry[3]],
                                        user_id_grade_dict[entry[0]] - float(entry[1]),
                                        business_id_grade_dict[entry[3]] - float(entry[1]),
                                        user_id_grade_dict[entry[0]] - business_id_grade_dict[entry[3]])]
    return value_dict


#makes lists for each difference that are easier to work with for correlations
#IMPORTANT FUNCTION, LOTS OF INFO
def grade_diffs():
    #unpack the values list
    stars_grades = [value for values in stars_grade_diff().values() for value in values]
    star_diff, user_grade_diff, business_grade_diff, user_business_diff = zip(*stars_grades)
    return [list(star_diff), list(user_grade_diff), list(business_grade_diff), list(user_business_diff)]


def diff_pair():
    diffs = grade_diffs()
    return [diffs[0], diffs[3]]


#trying to predict deviation from star average, so sort stars by grade index
def sort_diffs():
    star_diff = diff_pair()[0]
    grades_diff = diff_pair()[1]
    sorted_stars = [x for (y, x) in sorted(zip(grades_diff, star_diff))]
    return sorted_stars


#creates partitions of 5000 based on writing grade difference and star rating difference
def group_diffs():
    #stars sorted by corresponding grades
    sorted_stars = sort_diffs()
    sorted_grades = sorted(diff_pair()[1])
    grade_star_dict = {str(sorted_grades[4907]): sorted_stars[:4907]}
    for i in range(9907, len(sorted_stars), 5000):
        grade_star_dict[str(sorted_grades[i])] = sorted_stars[i-5000:i]
    return grade_star_dict


def partitions():
    return np.sort(list(group_diffs().keys()))


def partition_mean_std():
    partition_list = partitions()
    diff_dict = group_diffs()
    return {partition: (np.mean(diff_dict[partition]), np.std(diff_dict[partition]))
            for partition in partition_list}


#is the star rating affected by the difference between customer and business average writing grade?
#perhaps when the two are very similar the rating is good and when they diverge it gets worse?
#p = -0.0518x^2 + 0.1276x + 0.06336, which reaches max at about x = 1.1
#HYPOTHESIS TRUE
def diff_plots():
    diff_lists = grade_diffs()
    x = np.array(diff_lists[0])
    y = np.array(diff_lists[3])
    z = np.polyfit(x, y, 2)
    p = np.poly1d(z)
    xp = np.linspace(-5, 5, 100)
    plt.scatter(x, y)
    plt.xlim(-5, 5)
    plt.plot(xp, p(xp), '-')
    plt.ylabel('User writing level minus average business reviewer writing level')
    plt.xlabel('Grade given for the business minus the business\' average grade')
    plt.show()
    return p


def grade_categories():
    with open('grade_id_pairs.csv') as file:
        contents = csv.reader(file, delimiter=',')
        category_grade_dict = {}
        category_dict = id_categories()
        for entry in contents:
            business_id = entry[3]
            grade = entry[1]
            #get grades from ids
            try:
                for category in category_dict[business_id]:
                    try:
                        category_grade_dict[category].append(float(grade))
                    except KeyError:
                        category_grade_dict[category] = [float(grade)]
            except KeyError:
                break
    return category_grade_dict


#categories paired with their average grade level review
def grade_categories_avg():
    categories_grade_dict = grade_categories()
    categories = list(categories_grade_dict.keys())
    avg_dict = {}
    for category in categories:
        avg_dict[category] = sum(categories_grade_dict[category])/len(categories_grade_dict[category])
    return avg_dict


def predicted_business_rating():
    id_rating_dict = {}
    mean = fuc.mean_user_stars()
    category_ratings = average_category_rating()
    for entry in training_business_data:
        actual_rating = entry['stars']
        sum_expected_rating = sum([category_ratings[category] for category in categories(entry)])
        num_categories = len(categories(entry))
        review_count = entry['review_count']
        try:
            expected_rating = sum_expected_rating / num_categories
        except ZeroDivisionError:
            #gives more weight to their average rating if more reviews
            expected_rating = (mean + review_count*actual_rating)/(1 + review_count)
        predicted_rating = (actual_rating*review_count + expected_rating)/(review_count + 1)
        id_rating_dict[entry['business_id']] = predicted_rating
    for entry in test_business_data:
        try:
            predicted_rating = id_rating_dict[entry['business_id']]
        except KeyError:
            sum_expected_rating = sum([category_ratings[category] for category in categories(entry)
                                       if category in category_set(training_business_data)])
            num_categories = len(categories(entry))
            try:
                expected_rating = sum_expected_rating / num_categories
            except ZeroDivisionError:
                expected_rating = mean
            predicted_rating = expected_rating
            id_rating_dict[entry['business_id']] = predicted_rating
    return id_rating_dict



