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

checkpoint_dir = '../training/retrained_word_checkpoints'

print(tf.train.list_variables(tf.train.latest_checkpoint(checkpoint_dir)))
print(tf.train.latest_checkpoint(checkpoint_dir))

model = build_model(len(vocab), 256, 1024, batch_size=1)
model.load_weights(tf.train.latest_checkpoint(checkpoint_dir))
model.build(tf.TensorShape([1, None]))
model.summary()

def generate_text(model, start_string):
    num_generate = 200

    input_eval = [word2idx[word] for word in start_string.split(' ')]
    input_eval = tf.expand_dims(input_eval, 0)

    text_generated = []

    temperature = 1.0

    model.reset_states()
    for i in range(num_generate):
        predictions = model(input_eval)
        predictions = tf.squeeze(predictions, 0)

        predictions = predictions/temperature
        predicted_id = tf.random.categorical(predictions, num_samples=1)
        predicted_id = predicted_id[-1,0].numpy()

        input_eval = tf.expand_dims([predicted_id], 0)
        text_generated.append(idx2word[predicted_id])

    return (start_string + ' ' + ' '.join(text_generated))

print(generate_text(model, start_string=u"i am writing"))
