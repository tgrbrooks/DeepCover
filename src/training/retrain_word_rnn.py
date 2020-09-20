import tensorflow as tf
import numpy as np
import os
import time
import re

# Adapted from https://www.tensorflow.org/tutorials/text/text_generation
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

words_as_int = np.array([word2idx[c] for c in sentence_words])

# Maximum length sentence for single input in characters
seq_length = 30

word_dataset = tf.data.Dataset.from_tensor_slices(words_as_int)

sequences = word_dataset.batch(seq_length+1, drop_remainder=True)

def split_input_target(chunk):
    input_text = chunk[:-1]
    target_text = chunk[1:]
    return input_text, target_text

dataset = sequences.map(split_input_target)

BATCH_SIZE = 64
BUFFER_SIZE = 10000

dataset = dataset.shuffle(BUFFER_SIZE).batch(BATCH_SIZE, drop_remainder=True)

# Build the model

# Length of the vocabulary
vocab_size = len(vocab)
# Embedding dimension
embedding_dim = 256
# Number of RNN units
rnn_units = 1024

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

load_dir = './word_checkpoints'
model = build_model(vocab_size, embedding_dim, rnn_units, BATCH_SIZE)
model.load_weights(tf.train.latest_checkpoint(load_dir))

model.summary()
# Train the model

def loss(labels, logits):
    return tf.keras.losses.sparse_categorical_crossentropy(labels, logits, from_logits=True)

model.compile(optimizer='adam', loss=loss)

# Where the checkpoints will be saved
checkpoint_dir = './retrained_word_checkpoints'
# Name of the checkpoint files
checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt_{epoch}")

checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_prefix,
    save_weights_only=True)

EPOCHS = 20
history = model.fit(dataset, epochs=EPOCHS, callbacks=[checkpoint_callback])
