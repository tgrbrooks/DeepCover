import tensorflow as tf
import numpy as np
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

tokens = []
with open('../../data/raw_jobs.txt', 'r') as f:
    for line in f:
        if line.find('-->') != -1:
            continue
        tokens.append(re.sub(r"[^a-z.]+"," ", line.lower()))

text = " ".join(tokens)
text = text.replace('  ',' ')
text = text.replace('   ',' ')
print('Length of text: {} characters'.format(len(text)))

vocab = sorted(set(text))
print('{} unique characters'.format(len(vocab)))

# Map from unique characters to indices
char2idx = {u:i for i, u in enumerate(vocab)}
idx2char = np.array(vocab)

checkpoint_dir = '../training/retrained_checkpoints2'

print(tf.train.list_variables(tf.train.latest_checkpoint(checkpoint_dir)))
print(tf.train.latest_checkpoint(checkpoint_dir))

model = build_model(len(vocab), 256, 1024, batch_size=1)
model.load_weights(tf.train.latest_checkpoint(checkpoint_dir))
model.build(tf.TensorShape([1, None]))
model.summary()

def generate_text(model, start_string):
    num_generate = 1000

    input_eval = [char2idx[s] for s in start_string]
    #print(input_eval)
    input_eval = tf.expand_dims(input_eval, 0)
    #print(input_eval)

    text_generated = []

    temperature = 1.0

    model.reset_states()
    for i in range(num_generate):
        predictions = model(input_eval)
        #print(predictions)
        predictions = tf.squeeze(predictions, 0)

        predictions = predictions/temperature
        #print(predictions)
        predicted_id = tf.random.categorical(predictions, num_samples=1)
        #print(predicted_id)
        predicted_id = predicted_id[-1,0].numpy()
        #print(predicted_id)

        input_eval = tf.expand_dims([predicted_id], 0)
        #print(input_eval)
        text_generated.append(idx2char[predicted_id])

    return (start_string + ''.join(text_generated))

print(generate_text(model, start_string=u"i am writing"))
