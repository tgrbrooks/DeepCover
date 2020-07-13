import cv2
import pytesseract
from bs4 import BeautifulSoup
from io import BytesIO
import pytesseract
from spellchecker import SpellChecker
import re
import urllib.request, urllib.parse
import numpy as np
import os
import sys
import json
import ssl
import csv

# Function to return parsed HTML page from url
def get_soup(url, header):
    req = urllib.request.Request(url, headers=header)
    gcontext = ssl.SSLContext()
    return BeautifulSoup(urllib.request.urlopen(req, context=gcontext), 'html.parser')

# Initialise BytesIO object for reading images
image_stream = BytesIO()
# Initialise spell checker
spell = SpellChecker()

# Keep track of opened files
file_used = open('../../data/used_files.txt', 'r')
used_letters = []
lines = file_used.readlines()
for line in lines:
    used_letters.append(line.replace('\n',''))
file_used.close()

# Search for cover letters
base_query = 'cover letter'
# Directory to save results
save_directory = '/Users/tbrooks/Documents/DeepCover/data/'
# Specialised jobs
jobs = ["", "data scientist", "researcher", "physics", "software",
        "programmer", "software engineer", "analyst", "scientist"]
# Number of results to fetch at one time
num_per_query = 10
# Total number of queries for each job
num_queries = 100

# Loop over the job specialisations
for job in jobs:
    # Add to the query
    query = job + ' ' + base_query
    query = query.split()
    query = '+'.join(query)
    print('Searching for: '+job+' '+base_query)

    # Loop over the number of queries
    for query_i in range(0, num_queries):

        # Create image search url
        url = ("http://www.bing.com/images/search?q=" + 
              query + "&FORM=HDRSC2&first=" +
              str(query_i*10) + "&count=" +
              str(num_per_query))

        # Meta information
        header = {'User-Agent':"Mozilla/5.0 (Windows NT 6.1; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/43.0.2357.134 Safari/537.36"}

        # Return the parsed HTML document
        soup = get_soup(url, header)

        image_list = []
        num_used = 0
        file_used = open('../../data/used_files.txt', 'a')
        # Loop over the image results
        for a in soup.find_all("a", {"class":"iusc"}):
            # Fetch the high quality link
            m = json.loads(a["m"])
            turl = m["turl"]
            murl = m["murl"]

            # Get the name of the image
            image_name = urllib.parse.urlsplit(murl).path.split("/")[-1]
            print(image_name)

            # Check if image had been used before
            if image_name not in used_letters:
                image_list.append((image_name, turl, murl))
                used_letters.append(image_name)
                file_used.write(image_name)
                file_used.write('\n')
            else:
                print("File already used\n")
                num_used += 1
        file_used.close()

        # If all files were used, we're at the end of the search results
        if num_used == num_per_query:
            break

        # Loop over the high quality images
        for i, (image_name, turl, murl) in enumerate(image_list):
            # Protect against errors crashing the script
            try:
                # Tell the BytesIO object to go back to address 0
                image_stream.seek(0)

                # Get the image from the address
                raw_img = urllib.request.urlopen(turl).read()

                # Convert to a BytesIO object
                image_stream = BytesIO(raw_img)
                # Convert to a numpy array
                file_bytes = np.asarray(bytearray(image_stream.read()),
                                        dtype=np.uint8)

                # Use openCV to create image
                img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

                # Detect text from image with tesseract
                raw_text = pytesseract.image_to_string(img)
                text = raw_text[:]

                # Only interested in the main body of text
                # Find the start
                first = text.lower().find('dear')
                # If start found go to end of line, don't care about names
                if first != -1:
                    first = first + text[first:].lower().find('\n')

                # Find the end using common sign offs
                last = text.lower().find('sincerely')
                if last == -1:
                    last = text.lower().find('faithfully')
                if last == -1:
                    last = text.lower().find('best')

                # Remove all text above start and below end
                if first != -1 and last != -1:
                    first = first + 1
                    text = text[first:last]
                elif first != -1:
                    first = first + 1
                    text = text[first:]
                elif last != -1:
                    text = text[:last]

                # Replace common errors
                text = text.replace('\n', ' ')
                text = text.replace('|', 'I')

                # try to get sentences
                sentences_strings = text.split('.')
                sentences = []
                n_correct = 0
                n_incorrect = 0
                for sent_str in sentences_strings:
                    # tokenise and remove non alpha-numeric characters
                    tokens = re.sub(r"[^a-z0-9]+", " ", sent_str.lower()).split()
                    correct = spell.known(tokens)
                    incorrect = spell.unknown(tokens)
                    n_correct += len(correct)
                    n_incorrect += len(incorrect)

                    # Try to correct spelling mistakes from imperfect text
                    # detection
                    for i, token in enumerate(tokens):
                        tokens[i] = spell.correction(token)
                    # Remove too short sentences
                    if(len(tokens) > 2):
                        sentences.append(tokens)

                frac_incorrect = n_incorrect/(n_correct+n_incorrect)
                if frac_incorrect > 0.3:
                    print('Poor word detection, skipping...')
                    continue

                with open('../../data/raw_sentences.txt', 'a') as raw_file:
                    raw_file.write(raw_text)
            
                # Append to csv file
                with open('../../data/bing_sentences.csv', 'a') as csvfile:
                    writer = csv.writer(csvfile, delimiter=',')
                    for sentence in sentences:
                        writer.writerow(sentence)

            except Exception as e:
                print('Could not load: '+ image_name)
                print(e)
