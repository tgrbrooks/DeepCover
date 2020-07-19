import tensorflow as tf
import numpy as np
from spellchecker import SpellChecker
import csv
from collections import Counter

# Use trained model to fix dodgy sentences

spell = SpellChecker()

sentences = []
file_names = ["../../data/sentences.csv", "../../data/bing_sentences.csv"]
for file_name in file_names:
    print(file_name)
    with open(file_name, 'r') as csvfile:
        reader = csv.reader(csvfile, delimiter=',')
        count = 0
        for row in reader:
            print('Sentence '+str(count))
            count += 1
            '''
            row = [word.replace('\n','. ') for word in row]
            row = [word.replace('\r','') for word in row]
            row = [word.replace('asa','as a') for word in row]
            row = [word.replace('lam','i am') for word in row]
            row = [word.replace('tam','i am') for word in row]
            row = [word.replace('nd','and') for word in row]
            row = [word.replace('ofthe','of the') for word in row]
            row = [word.replace('forth','for the') for word in row]
            row = [word.replace('ae','are') for word in row]
            row = [word.replace('fr','for') for word in row]
            row = [word.replace('wil','will') for word in row]
            row = [word.replace('ad','and') for word in row]
            row = [word.replace('inthe','in the') for word in row]
            row = [word.replace('fo','for') for word in row]
            row = [word.replace('wth','with') for word in row]
            row = [word.replace('al','all') for word in row]
            row = [word.replace('ny','any') for word in row]
            row = [word.replace('th','the') for word in row]
            row = [word.replace('re','are') for word in row]
            row = [word.replace('ca','can') for word in row]
            '''
            for i, word in enumerate(row):
                row[i] = spell.correction(word)
            n_correct = len(spell.known(row))
            n_incorrect = len(spell.unknown(row))
            if n_correct+n_incorrect == 0:
                continue
            frac_incorrect = n_incorrect/(n_correct+n_incorrect)
            if frac_incorrect < 0.2:
                sentences.append(row)
            else:
                print("Sentence rejected: ",row)
                print(n_incorrect, n_correct, frac_incorrect)

with open('../../data/cleaned_sentences.csv', 'w') as csvfile:
    writer = csv.writer(csvfile, delimiter=',')
    for sentence in sentences:
        writer.writerow(sentence)
