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

'''
text = open('../../data/sentences.csv', 'rb').read().decode(encoding='utf-8').replace(',',' ')
text = text.replace('\n','. ')
text = text.replace('\r','')
text = text.replace('/','')
text = text.replace('-','')
text = text.replace('=','')
#text = text.replace('asa','as a')
'''

tokens = []
with open('../../data/raw_jobs.txt', 'r') as f:
    for line in f:
        if line.find('-->') != -1:
            continue
        tokens.append(re.sub(r"[^a-z.]+"," ", line.lower()))

text = " ".join(tokens)
text = text.replace('  ',' ')
text = text.replace('   ',' ')

vocab = sorted(set(text))

# Map from unique characters to indices
char2idx = {u:i for i, u in enumerate(vocab)}
idx2char = np.array(vocab)

checkpoint_dir = '../training/retrained_checkpoints2'

model = build_model(len(vocab), 256, 1024, batch_size=1)
model.load_weights(tf.train.latest_checkpoint(checkpoint_dir))
model.build(tf.TensorShape([1, None]))
model.summary()
parsed = json.loads(model.to_json())
print(json.dumps(parsed, indent=2, sort_keys=False))
print(model.get_weights()[0])

tfjs.converters.save_keras_model(model, 'retrained_model/model')

def generate_text(model, start_string):
    num_generate = 1

    input_eval = [char2idx[s] for s in start_string]
    input_eval = tf.expand_dims(input_eval, 0)
    print('input = ',input_eval)

    text_generated = []

    temperature = 1.0

    #model.reset_states()
    for i, layer in enumerate(model.layers):
        output = layer(input_eval)
        print('layer',i,': ',output)
        input_eval = output

    '''
    for i in range(num_generate):
        predictions = model(input_eval)
        print(predictions)
        predictions = tf.squeeze(predictions, 0)
        #print(predictions)

        predictions = predictions/temperature
        #print(predictions)
        predicted_id = tf.random.categorical(predictions, num_samples=1)
        #print(predicted_id)
        predicted_id = predicted_id[-1,0].numpy()
        #print(predicted_id)

        input_eval = tf.expand_dims([predicted_id], 0)
        #print(input_eval)
        #print(idx2char[predicted_id])
        text_generated.append(idx2char[predicted_id])
    '''

    return (start_string + ''.join(text_generated))

print(generate_text(model, start_string=u"a"))
