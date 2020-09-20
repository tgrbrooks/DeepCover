import tensorflow as tf
import tensorflowjs as tfjs
import numpy as np
import json
import re

def build_model(vocab_size, embedding_dim, rnn_units, batch_size):
    model = tf.keras.Sequential([
        tf.keras.layers.Embedding(vocab_size, embedding_dim,
                                  batch_input_shape=[batch_size, None]),
        tf.keras.layers.GRU(rnn_units, return_sequences=True, stateful=True,
                            recurrent_initializer='glorot_uniform',
                            reset_after=False),
        tf.keras.layers.Dense(vocab_size)
    ])
    return model

text = open('../../data/jobs.txt', 'rb').read().decode(encoding='utf-8')
text = text.replace('\n', ' ')
words = text.split(' ')

sentence_text = open('../../data/sentences.txt', 'rb').read().decode(encoding='utf-8')
sentence_text = sentence_text.replace('\n', ' ')
sentence_words = sentence_text.split(' ')

vocab = sorted(set(words).union(set(sentence_words)))
vocab_size = len(vocab)

word2idx = {u:i for i, u in enumerate(vocab)}
idx2word = np.array(vocab)

with open('word2idx.js', 'w') as f:
    f.write('let word2idx = new Map([\n')
    for i, word in enumerate(vocab):
        f.write('  ["'+word+'", '+str(i)+']')
        if i == len(vocab)-1:
            f.write('\n')
        else:
            f.write(',\n')
    f.write('])')       

checkpoint_dir = '../training/retrained_word_checkpoints'

model = build_model(len(vocab), 256, 1024, batch_size=1)
model.load_weights(tf.train.latest_checkpoint(checkpoint_dir))
model.build(tf.TensorShape([1, None]))
model.summary()
parsed = json.loads(model.to_json())

model.save('word_model.h5')
tfjs.converters.save_keras_model(model, 'word_model/model')
