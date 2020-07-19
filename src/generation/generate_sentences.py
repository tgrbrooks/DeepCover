import tensorflow as tf
import numpy as np
from spellchecker import SpellChecker

spell = SpellChecker()

skills = ['Research', 'Engineer', 'Developer', 'Python', 'Programming', 'PhD', 'Publications']

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

vocab = sorted(set(text))

# Map from unique characters to indices
char2idx = {u:i for i, u in enumerate(vocab)}
idx2char = np.array(vocab)

forwards_dir = '../training/training_checkpoints3'
backwards_dir = '../training/backwards_checkpoints'

forwards_model = build_model(len(vocab), 256, 1024, batch_size=1)
forwards_model.load_weights(tf.train.latest_checkpoint(forwards_dir))
forwards_model.build(tf.TensorShape([1, None]))

backwards_model = build_model(len(vocab), 256, 1024, batch_size=1)
backwards_model.load_weights(tf.train.latest_checkpoint(backwards_dir))
backwards_model.build(tf.TensorShape([1, None]))

def generate_sentence(forwards_model, backwards_model, seed_word):
    num_generate = 400
    temperature = 1.0

    # Forwards pass
    input_eval = [char2idx[s] for s in seed_word]
    input_eval = tf.expand_dims(input_eval, 0)

    accepted = False
    count = 0

    while not accepted:
        # If it can't generate anything good return nothing
        if count > 10:
            return ''

        sentence = []
        text_generated = []
        forwards_model.reset_states()
        for i in range(num_generate):
            predictions = forwards_model(input_eval)
            predictions = tf.squeeze(predictions, 0)

            predictions = predictions/temperature
            predicted_id = tf.random.categorical(predictions, num_samples=1)[-1,0].numpy()

            input_eval = tf.expand_dims([predicted_id], 0)
            text_generated.append(idx2char[predicted_id])
        sentence = (seed_word + ''.join(text_generated))
        full_stop = True
        if len(sentence.split('.')) == 1:
            full_stop = False
        sentence = sentence.split('.')[0]
        words = sentence.split(' ')
        num_correct = len(spell.known(words))
        num_incorrect = len(spell.unknown(words))
        frac_correct = num_correct/(num_correct+num_incorrect)
        if frac_correct >= 0.8: 
            accepted = True
        if len(sentence) < 5 or len(sentence) > 150:
            accepted = False
        if not full_stop:
            accepted = False
        count += 1

    # If the seed is the start the start of the sentence
    if seed_word[0] != ' ':
        return sentence

    # Backwards pass
    seed = sentence[::-1]
    input_eval = [char2idx[s] for s in seed]
    input_eval = tf.expand_dims(input_eval, 0)

    accepted = False
    count = 0

    while not accepted:
        # If it can't generate anything good return nothing
        if count > 10:
            return ''

        sentence = []
        text_generated = []
        backwards_model.reset_states()
        for i in range(num_generate):
            predictions = backwards_model(input_eval)
            predictions = tf.squeeze(predictions, 0)

            predictions = predictions/temperature
            predicted_id = tf.random.categorical(predictions, num_samples=1)[-1,0].numpy()

            input_eval = tf.expand_dims([predicted_id], 0)
            text_generated.append(idx2char[predicted_id])
        sentence = (seed + ''.join(text_generated))
        full_stop = True
        if len(sentence.split('.')) == 1:
            full_stop = False
        sentence = sentence.split('.')[0]
        sentence = sentence[::-1]
        words = sentence.split(' ')
        num_correct = len(spell.known(words))
        num_incorrect = len(spell.unknown(words))
        frac_correct = num_correct/(num_correct+num_incorrect)
        if frac_correct >= 0.8: 
            accepted = True
        if len(sentence) < 50 or len(sentence) > 300:
            accepted = False
        if not full_stop:
            accepted = False
        count += 1

    return sentence

letter_body = ['I am writing to apply for the position of data scientist at your company.']

for skill in skills:
    skill = (' '+skill+' ').lower()
    sentence = generate_sentence(forwards_model,
                                 backwards_model,
                                 skill)
    sentence = sentence[1:]
    words = sentence.split(' ')
    corrected_sentence = []
    for word in words:
        if word == ' ':
            continue
        corrected_word = spell.correction(word)
        if len(spell.known([corrected_word]))==1 or word.isnumeric():
            corrected_sentence.append(corrected_word)

    sentence = ' '.join(corrected_sentence) + '.'
    sentence = sentence[0].upper() + sentence[1:]
    sentence = sentence.replace(' i ',' I ')
    letter_body.append(sentence)

for sent in letter_body:
    print(sent)
