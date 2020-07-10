# Imports
from google_images_search import GoogleImagesSearch
from io import BytesIO
import cv2
import numpy as np
import pytesseract
from spellchecker import SpellChecker
import re
import csv
import os
import sys

# Initialise BytesIO object for reading images
image_stream = BytesIO()
# Initialise spell checker
spell = SpellChecker()
# Initialise google image search (uses API)
gis = GoogleImagesSearch(os.environ['GCS_DEVELOPER_KEY'],
                         os.environ['GCS_CX'])

# Run the search, protecting against failures crashing the script
failed = True
# Keep count of the failures, many probably means an actual issue rather than
# one dodgy image
fail_count = 0
while failed:
    try:
        # Search for user input, 10 images at a time
        # Looking for letters, so want white background
        gis.search({'q': sys.argv[1],
                    'num': 10,
                    'start': 1,
                    'safe': 'high',
                    'fileType': 'jpg',
                    'imgDominantColor': 'white'})
        failed = False
    except Exception as e:
        failed = True
        fail_count += 1
        if fail_count > 10:
            raise RuntimeError(e)

# Big loop to keep getting images until daily API quota reached
for i in range(0, 2000):
    # Loop over the image search results
    for ind, image in enumerate(gis.results()):
        # Protect against errors crashing the script
        try:
            print('Image :' + str(ind))
            # here we tell the BytesIO object to go back to address 0
            image_stream.seek(0)

            # take raw image data
            raw_image_data = image.get_raw_data()

            # this function writes the raw image data to the object
            image.copy_to(image_stream, raw_image_data)

            # Back to address 0 to write to numpy array
            image_stream.seek(0)
            file_bytes = np.asarray(bytearray(image_stream.read()), dtype=np.uint8)

            # Use openCV to create image
            img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

            # Detect text from image with tesseract
            text = pytesseract.image_to_string(img)

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
            for sent_str in sentences_strings:
                # tokenise and remove non alpha-numeric characters
                tokens = re.sub(r"[^a-z0-9]+", " ", sent_str.lower()).split()
                # Try to correct spelling mistakes from imperfect text
                # detection
                for i, token in enumerate(tokens):
                    tokens[i] = spell.correction(token)
                # Remove too short sentences
                if(len(tokens) > 2):
                    sentences.append(tokens)
            
            # Append to csv file
            with open('../../data/sentences.csv', 'a') as csvfile:
                writer = csv.writer(csvfile, delimiter=',')
                for sentence in sentences:
                    writer.writerow(sentence)
        # Ignore errors, go to next image
        except Exception:
            pass

    # At each loop iteration, go to next page of results
    # Protect against errors as before
    next_failed = True
    next_fail_count = 0
    while next_failed:
        try:
            gis.next_page()
            next_failed = False
        except Exception as e:
            next_failed = True
            next_fail_count += 1
            if next_fail_count > 10:
                raise RuntimeError(e)
