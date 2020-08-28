import tensorflow as tf
import numpy as np
import os
import time
import re

# Adapted from https://www.tensorflow.org/tutorials/text/text_generation
text = open('../../data/sentences.csv', 'rb').read().decode(encoding='utf-8').replace(',',' ')
text = re.sub(r"[^a-z.]+"," ",text)
text = text.replace('\n','. ')
text = text.replace('\r','')
text = text.replace('/','')
text = text.replace('-','')
text = text.replace('=','')

vocab = sorted(set(text))
print('{} unique characters'.format(len(vocab)))

# Map from unique characters to indices
char2idx = {u:i for i, u in enumerate(vocab)}
idx2char = np.array(vocab)

# Vectorise the text
text_as_int = np.array([char2idx[c] for c in text])

# Print the mapping
print('{')
for char,_ in zip(char2idx, range(len(vocab))):
    print(' {:4s}: {:3d},'.format(repr(char), char2idx[char]))
print('}')

# Maximum length sentence for single input in characters
seq_length = 100
examples_per_epoch = len(text)

# Create training examples/targets
char_dataset = tf.data.Dataset.from_tensor_slices(text_as_int)

# Convert individual characters to sequences of the desired size
sequences = char_dataset.batch(seq_length+1, drop_remainder=True)

# Duplicate and shift each sequence to form input and target
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

load_dir = './job_checkpoints2'
model = build_model(vocab_size, embedding_dim, rnn_units, BATCH_SIZE)
model.load_weights(tf.train.latest_checkpoint(load_dir))

model.summary()
# Train the model

def loss(labels, logits):
    return tf.keras.losses.sparse_categorical_crossentropy(labels, logits, from_logits=True)

model.compile(optimizer='adam', loss=loss)

# Where the checkpoints will be saved
checkpoint_dir = './retrained_checkpoints2'
# Name of the checkpoint files
checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt_{epoch}")

checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_prefix,
    save_weights_only=True)

EPOCHS = 20
history = model.fit(dataset, epochs=EPOCHS, callbacks=[checkpoint_callback])
