import tensorflow as tf
import numpy as np
import os
import time

# Adapted from https://www.tensorflow.org/tutorials/text/text_generation

# Create the dataset

text = open('../../data/sentences.csv', 'rb').read().decode(encoding='utf-8').replace(',',' ')
text = text.replace('\n','. ')
text = text.replace('\r','')
text = text.replace('asa','as a')
# Flip the text
text = text[::-1]
print('Length of text: {} characters'.format(len(text)))

vocab = sorted(set(text))
print('{} unique characters'.format(len(vocab)))
print(text[:250])
print(text[:250][::-1])

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

for i in char_dataset.take(5):
    print(idx2char[i.numpy()])

# Convert individual characters to sequences of the desired size
sequences = char_dataset.batch(seq_length+1, drop_remainder=True)

for item in sequences.take(5):
    print(repr(''.join(idx2char[item.numpy()])))

# Duplicate and shift each sequence to form input and target
def split_input_target(chunk):
    input_text = chunk[:-1]
    target_text = chunk[1:]
    return input_text, target_text

dataset = sequences.map(split_input_target)

for input_example, target_example in dataset.take(1):
    print('Input data:', repr(''.join(idx2char[input_example.numpy()])))
    print('Input data (flipped):', repr(''.join(idx2char[input_example.numpy()]))[::-1])
    print('Target data:', repr(''.join(idx2char[target_example.numpy()])))
    print('Target data (flipped):', repr(''.join(idx2char[target_example.numpy()]))[::-1])

for i, (input_idx, target_idx) in enumerate(zip(input_example[:5], target_example[:5])):
    print("Step {:4d}".format(i))
    print(" input: {} ({:s})".format(input_idx, repr(idx2char[input_idx])))
    print(" expected output: {} ({:s})".format(target_idx, repr(idx2char[target_idx])))

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
                            recurrent_initializer='glorot_uniform'),
        tf.keras.layers.Dense(vocab_size)
    ])
    return model

model = build_model(vocab_size, embedding_dim, rnn_units, BATCH_SIZE)

for input_example_batch, target_example_batch in dataset.take(1):
    example_batch_predictions = model(input_example_batch)
    print(example_batch_predictions.shape, '# (batch_size, sequence_length, vocab_size)')

model.summary()

sampled_indices = tf.random.categorical(example_batch_predictions[0], num_samples=1)
sampled_indices = tf.squeeze(sampled_indices, axis=-1).numpy()

print("Input: \n", repr("".join(idx2char[input_example_batch[0]])))
print("Input (flipped): \n", repr("".join(idx2char[input_example_batch[0]]))[::-1])
print("Next char predictions: \n", repr("".join(idx2char[sampled_indices])))

# Train the model

def loss(labels, logits):
    return tf.keras.losses.sparse_categorical_crossentropy(labels, logits, from_logits=True)

example_batch_loss = loss(target_example_batch, example_batch_predictions)
print("Prediction shape: ", example_batch_predictions.shape, " # (batch_size, sequence_length, vocab_size)")
print("scalar_loss:      ", example_batch_loss.numpy().mean())

model.compile(optimizer='adam', loss=loss)

# Where the checkpoints will be saved
checkpoint_dir = './backwards_checkpoints'
# Name of the checkpoint files
checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt_{epoch}")

checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_prefix,
    save_weights_only=True)

EPOCHS = 20
history = model.fit(dataset, epochs=EPOCHS, callbacks=[checkpoint_callback])
