__author__ = 'slyfocks'
import json
import numpy as np

with open('yelp_test_set_checkin.json') as file:
    checkin_data = [json.loads(entry) for entry in file]

