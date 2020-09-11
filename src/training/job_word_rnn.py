import tensorflow as tf
import tensorflow_datasets as tfds
import numpy as np
import os
import time
import re

# Adapted from https://www.tensorflow.org/tutorials/text/text_generation

def labeler(example, index):
    return example, tf.cast(index, tf.int64)

lines_dataset = tf.data.TextLineDataset('../../data/jobs.txt')
label_dataset = lines_dataset.map(lambda ex: labeler(ex, 0))

for ex in lines_dataset.take(1):
    print(ex)

tokenizer = tfds.features.text.Tokenizer()
vocabulary_set = set()

for text_tensor,_ in label_dataset:
  some_tokens = tokenizer.tokenize(text_tensor.numpy())
  vocabulary_set.update(some_tokens)

vocab_size = len(vocabulary_set)
print(vocab_size)

encoder = tfds.features.text.TokenTextEncoder(vocabulary_set)

example_text = next(iter(label_dataset))[0].numpy()
print(example_text)
encoded_example = encoder.encode(example_text)
print(encoded_example)

def encode(text_tensor, label):
  encoded_text = encoder.encode(text_tensor.numpy())
  return encoded_text, label

def encode_map_fn(text, label):
  # py_func doesn't set the shape of the returned tensors.
  encoded_text, label = tf.py_function(encode, inp=[text, label], Tout=(tf.int64, tf.int64))

  # `tf.data.Datasets` work best if all components have a shape set
  #  so set the shapes manually: 
  encoded_text.set_shape([None])
  label.set_shape([])

  return encoded_text, label

encoded_data = label_dataset.map(encode_map_fn)
for ex in encoded_data.take(1):
    print(ex)

# Maximum length sentence for single input in characters
seq_length = 10

# Convert individual characters to sequences of the desired size
sequences = encoded_data.padded_batch(seq_length+1, drop_remainder=True)

for item in sequences.take(1):
    print(item)
'''
vocab_size += 1

# Duplicate and shift each sequence to form input and target
def split_input_target(chunk, label):
    input_text = chunk[:-1]
    target_text = chunk[1:]
    return input_text, target_text

dataset = sequences.map(split_input_target)

for item in dataset.take(5):
    print(item)

BATCH_SIZE = 64
BUFFER_SIZE = 10000

dataset = dataset.shuffle(BUFFER_SIZE).batch(BATCH_SIZE, drop_remainder=True)

# Build the model

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

model = build_model(vocab_size, embedding_dim, rnn_units, BATCH_SIZE)

model.summary()

# Train the model
def loss(labels, logits):
    return tf.keras.losses.sparse_categorical_crossentropy(labels, logits, from_logits=True)

model.compile(optimizer='adam', loss=loss)

# Where the checkpoints will be saved
checkpoint_dir = './word_checkpoints'
# Name of the checkpoint files
checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt_{epoch}")

checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_prefix,
    save_weights_only=True)


EPOCHS = 1
history = model.fit(dataset, epochs=EPOCHS, callbacks=[checkpoint_callback])
'''
