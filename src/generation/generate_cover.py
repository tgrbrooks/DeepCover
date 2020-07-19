import tensorflow as tf
import numpy as np

def build_model(vocab_size, embedding_dim, rnn_units, batch_size):
    model = tf.keras.Sequential([
        tf.keras.layers.Embedding(vocab_size, embedding_dim,
                                  batch_input_shape=[batch_size, None]),
        tf.keras.layers.GRU(rnn_units, return_sequences=True, stateful=True,
                            recurrent_initializer='glorot_uniform'),
        tf.keras.layers.Dense(vocab_size)
    ])
    return model

text = open('../../data/sentences.csv', 'rb').read().decode(encoding='utf-8').replace(',',' ')
text = text.replace('\n','. ')
text = text.replace('\r','')
text = text.replace('asa','as a')
'''
text = text.replace('lam ','i am ')
text = text.replace('tam ','i am ')
text = text.replace(' nd ',' and ')
text = text.replace(' ofthe ',' of the ')
text = text.replace(' forth ',' for the ')
text = text.replace(' ae ',' are ')
text = text.replace(' fr ',' for ')
text = text.replace(' wil ',' will ')
text = text.replace(' ad ',' and ')
text = text.replace(' inthe ',' in the ')
text = text.replace(' fo ',' for ')
text = text.replace(' wth ',' with ')
text = text.replace(' al ',' all ')
text = text.replace(' ny ',' any ')
text = text.replace(' th ',' the ')
text = text.replace(' re ',' are ')
text = text.replace(' ca ',' can ')
'''
print('Length of text: {} characters'.format(len(text)))

vocab = sorted(set(text))
#vocab.remove('=')
print('{} unique characters'.format(len(vocab)))

# Map from unique characters to indices
char2idx = {u:i for i, u in enumerate(vocab)}
idx2char = np.array(vocab)

checkpoint_dir = '../training/training_checkpoints3'

print(tf.train.list_variables(tf.train.latest_checkpoint(checkpoint_dir)))
print(tf.train.latest_checkpoint(checkpoint_dir))

model = build_model(len(vocab), 256, 1024, batch_size=1)
model.load_weights(tf.train.latest_checkpoint(checkpoint_dir))
model.build(tf.TensorShape([1, None]))

model.summary()

def generate_text(model, start_string):
    num_generate = 1000

    input_eval = [char2idx[s] for s in start_string]
    input_eval = tf.expand_dims(input_eval, 0)

    text_generated = []

    temperature = 1.0

    model.reset_states()
    for i in range(num_generate):
        predictions = model(input_eval)
        predictions = tf.squeeze(predictions, 0)

        predictions = predictions/temperature
        predicted_id = tf.random.categorical(predictions, num_samples=1)[-1,0].numpy()

        input_eval = tf.expand_dims([predicted_id], 0)
        text_generated.append(idx2char[predicted_id])

    return (start_string + ''.join(text_generated))

print(generate_text(model, start_string=u"i am applying"))
