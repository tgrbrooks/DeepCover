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

def get_soup(url, header):
    req = urllib.request.Request(url,headers=header)
    gcontext = ssl.SSLContext()
    return BeautifulSoup(urllib.request.urlopen(req, context=gcontext), 'html.parser')

image_stream = BytesIO()
spell = SpellChecker()

# Keep track of opened files
file_used = open('../../data/used_files.txt', 'r')
used_letters = []
lines = file_used.readlines()
for line in lines:
    used_letters.append(line.replace('\n',''))
file_used.close()

# Put user input into variables
base_query = 'cover letter'
max_images = 10
save_directory = '/Users/tbrooks/Documents/DeepCover/data/'
jobs = ["", "data scientist", "researcher", "physics", "software",
        "programmer", "software engineer", "analyst", "scientist"]
num_per_query = 10
num_queries = 100

for job in jobs:
    query = job + ' ' + base_query
    query = query.split()
    query = '+'.join(query)
    for query_i in range(0, num_queries):

        # Create google images url
        url="http://www.bing.com/images/search?q="+query+"&FORM=HDRSC2&first="+str(query_i*10)+"&count="+str(num_per_query)

        # Something to do with browsers
        header = {'User-Agent':"Mozilla/5.0 (Windows NT 6.1; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/43.0.2357.134 Safari/537.36"}
        soup = get_soup(url, header)

        image_list = []
        file_used = open('../../data/used_files.txt', 'a')
        for a in soup.find_all("a", {"class":"iusc"}):
            m = json.loads(a["m"])
            turl = m["turl"]
            murl = m["murl"]

            image_name = urllib.parse.urlsplit(murl).path.split("/")[-1]
            print(image_name)
            if image_name not in used_letters:
                image_list.append((image_name, turl, murl))
                file_used.write(image_name)
                file_used.write('\n')
            else:
                print("File already used\n")
        file_used.close()

        for i, (image_name, turl, murl) in enumerate(image_list):
            try:
                image_stream.seek(0)
                raw_img = urllib.request.urlopen(turl).read()
                image_stream = BytesIO(raw_img)
                file_bytes = np.asarray(bytearray(image_stream.read()),
                                        dtype=np.uint8)
                img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
                text = pytesseract.image_to_string(img)

                check_text = text.replace('\n', '')
                sentences = text.split('.')
                n_correct = 0
                n_incorrect = 0
                for sentence in sentences:
                    tokens = sentence.lower().split()
                    correct = spell.known(tokens)
                    incorrect = spell.unknown(tokens)
                    n_correct += len(correct)
                    n_incorrect += len(incorrect)
                    
                frac_incorrect = n_incorrect/(n_correct+n_incorrect)
                if frac_incorrect > 0.3:
                    print('Poor word detection, skipping...')
                    continue

                with open('../../data/raw_sentences.txt', 'a') as raw_file:
                    raw_file.write(text)

            except Exception as e:
                print('Could not load: '+ image_name)
                print(e)
