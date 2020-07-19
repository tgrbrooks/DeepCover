import numpy as np
import os
from random import shuffle
import re
from bokeh.models import ColumnDataSource, LabelSet
from bokeh.plotting import figure, show, output_file
from bokeh.io import output_notebook
import urllib.request
import zipfile
import lxml.etree

# Download the dataset if it's not already there: this may take a minute as it is 75MB
if not os.path.isfile('ted_en-20160408.zip'):
    urllib.request.urlretrieve("https://wit3.fbk.eu/get.php?path=XML_releases/xml/ted_en-20160408.zip&filename=ted_en-20160408.zip", filename="ted_en-20160408.zip")

# For now, we're only interested in the subtitle text, so let's extract that from the XML:
with zipfile.ZipFile('ted_en-20160408.zip', 'r') as z:
    doc = lxml.etree.parse(z.open('ted_en-20160408.xml', 'r'))
input_text = '\n'.join(doc.xpath('//content/text()'))
del doc

i = input_text.find("Hyowon Gweon: See this?")
input_text[i-20:i+150]

input_text_noparens = re.sub(r'\([^)]*\)', '', input_text)

i = input_text_noparens.find("Hyowon Gweon: See this?")
input_text_noparens[i-20:i+150]

sentences_strings_ted = []
for line in input_text_noparens.split('\n'):
    m = re.match(r'^(?:(?P<precolon>[^:]{,20}):)?(?P<postcolon>.*)$', line)
    sentences_strings_ted.extend(sent for sent in m.groupdict()['postcolon'].split('.') if sent)

# Uncomment if you need to save some RAM: these strings are about 50MB.
# del input_text, input_text_noparens

# Let's view the first few:
sentences_strings_ted[:5]

sentences_ted = []
for sent_str in sentences_strings_ted:
    tokens = re.sub(r"[^a-z0-9]+", " ", sent_str.lower()).split()
    sentences_ted.append(tokens)

len(sentences_ted)

print(sentences_ted[0])
print(sentences_ted[1])

# ...
words_ted = []
for sen in sentences_ted:
    words_ted.extend(sen)
    
from collections import Counter

c = Counter(words_ted)
counts_ted_top1000 = []
words_top_ted = []
for count in c.most_common(1000):
    counts_ted_top1000.append(count[1])
    words_top_ted.append(count[0])

from gensim.models import Word2Vec

model_ted = Word2Vec(sentences_ted, min_count=10)
print(len(model_ted.wv.vocab))

# This assumes words_top_ted is a list of strings, the top 1000 words
words_top_vec_ted = model_ted.wv[words_top_ted]

from sklearn.manifold import TSNE
tsne = TSNE(n_components=2, random_state=0)
words_top_ted_tsne = tsne.fit_transform(words_top_vec_ted)
