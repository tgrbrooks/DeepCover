import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from spellchecker import SpellChecker

spell = SpellChecker()

text = open('../../data/sentences.csv', 'rb').read().decode(encoding='utf-8').replace(',',' ')
text = text.replace('\n','. ')
text = text.replace('\r','')
text = text.replace('asa','as a')

vocab = sorted(set(text))

# Map from unique characters to indices
char2idx = {u:i for i, u in enumerate(vocab)}
idx2char = np.array(vocab)

def build_model(vocab_size, embedding_dim, rnn_units, batch_size):
    model = tf.keras.Sequential([
        tf.keras.layers.Embedding(vocab_size, embedding_dim,
                                  batch_input_shape=[batch_size, None]),
        tf.keras.layers.GRU(rnn_units, return_sequences=True, stateful=True,
                            recurrent_initializer='glorot_uniform'),
        tf.keras.layers.Dense(vocab_size)
    ])
    return model

def test_generation(model, start_string, backwards=False):
    num_generate = 1000

    if backwards:
        start_string = start_string[::-1]
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

    words = (start_string + ''.join(text_generated))
    if backwards:
        words = words[::-1]
    words = words.split(' ')
    num_correct = len(spell.known(words))
    num_incorrect = len(spell.unknown(words))
    frac_correct = num_correct/(num_correct+num_incorrect)
    return frac_correct

checkpoint_dir = './backwards_checkpoints'
num_checkpoints = 20
x = []
y = []
for i in range(1, num_checkpoints+1):
    model_path = checkpoint_dir + '/ckpt_' + str(i)
    print(model_path)

    model = build_model(len(vocab), 256, 1024, batch_size=1)
    model.load_weights(model_path)
    model.build(tf.TensorShape([1, None]))

    x.append(i)
    y.append(test_generation(model, start_string=u"i am applying", backwards=True))

plt.plot(x, y)
plt.xlabel('Checkpoint')
plt.ylabel('Correct word fraction')
plt.show()
