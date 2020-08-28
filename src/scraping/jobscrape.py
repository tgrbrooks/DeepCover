from selenium import webdriver
from selenium.webdriver.common.keys import Keys
import numpy as np
import re
import os
import sys
import json
import ssl
import csv

driver = webdriver.Firefox()
driver.implicitly_wait(5) # seconds

url_base = 'https://www.indeed.co.uk/jobs?q='
location = '&l=London%2C+Greater+London&start='
#jobs = ["", "data scientist", "researcher", "physics", "software",
#        "programmer", "software engineer", "analyst", "scientist"]
jobs = ['data_scientist']

# Directory to save results
save_directory = '/Users/tbrooks/Documents/DeepCover/data/'
# Number of results to fetch at one time
num_per_query = 10
# Total number of queries for each job
num_queries = 100

raw_file = open('../../data/raw_jobs.txt', 'a')

# Loop over the job specialisations
for job in jobs:
    # Add to the query
    query = job.split()
    query = '+'.join(query)

    # Loop over the number of queries
    for query_i in range(0, num_queries):

        # Create image search url
        url = (url_base + query + location + str(query_i*num_per_query))

        # Return the parsed HTML document
        driver.get(url)
        elements = driver.find_elements_by_class_name("clickcard")
        for element in elements:
            try:
                element.click()
                title = driver.find_element_by_id("vjs-jobtitle").text
                body = driver.find_element_by_id("vjs-desc").text
                raw_file.write('--->Job Title\n')
                raw_file.write(title+'\n')
                raw_file.write('--->Description\n')
                raw_file.write(body+'\n')
            except:
                continue

driver.close()
raw_file.close()
