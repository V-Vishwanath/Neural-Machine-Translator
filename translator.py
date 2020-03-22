import re
import os
import time
import numpy as np
import pandas as pd
import tensorflow as tf
from string import digits
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Layer, Embedding, GRU, Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import SparseCategoricalCrossentropy as SCC

class Translator:
    class _Encoder(tf.keras.Model):
        def __init__(self, size, embedding_dim, units, batch_size):
            super(Translator._Encoder, self).__init__()
            self.batch_size = batch_size
            self.units = units
            self.embedding = Embedding(size, embedding_dim)
            self.gru = GRU(self.units, return_sequences=True, return_state=True, recurrent_initializer='glorot_uniform')

        def call(self, x, hidden):
            x = self.embedding(x)
            return self.gru(x, initial_state = hidden)

        def init_hidden(self):
            return tf.zeros((self.batch_size, self.units))


    class _Attention(tf.keras.layers.Layer):
        def __init__(self, units):
            super(Translator._Attention, self).__init__()
            self.X = Dense(units)
            self.Y = Dense(units)
            self.V = Dense(1)

        def call(self, query, values):
            score = self.V(tf.nn.tanh(self.X(tf.expand_dims(query, 1)) + self.Y(values)))
            attention_weights = tf.nn.softmax(score, axis=1)
            context_vector = attention_weights * values
            context_vector = tf.reduce_sum(context_vector, axis=1)
            return context_vector, attention_weights


    class _Decoder(tf.keras.Model):
        def __init__(self, size, embedding_dim, units, batch_size):
            super(Translator._Decoder, self).__init__()
            self.batch_size = batch_size
            self.units = units
            self.embedding = tf.keras.layers.Embedding(size, embedding_dim)
            self.gru = tf.keras.layers.GRU(self.units, return_sequences=True, return_state=True, recurrent_initializer='glorot_uniform')
            self.fc = tf.keras.layers.Dense(size)
            self.attention = Translator._Attention(self.units)

        def call(self, x, hidden, enc_output):
            context_vector, attention_weights = self.attention(hidden, enc_output)
            x = self.embedding(x)
            x = tf.concat([tf.expand_dims(context_vector, 1), x], axis=-1)
            output, state = self.gru(x)
            output = tf.reshape(output, (-1, output.shape[2]))
            return self.fc(output), state, attention_weights


    def __init__(self, inp_lang_samples, out_lang_samples):
        inp_lang_samples = [self._preprocess(i) for i in inp_lang_samples]
        out_lang_samples = [self._preprocess(i) for i in out_lang_samples]

        inp_tensors, self._inp_tokenizer = self._tokenize(inp_lang_samples)
        out_tensors, self._out_tokenizer = self._tokenize(out_lang_samples)
        self.max_inp_len = max(len(tensor) for tensor in inp_tensors)
        self.max_out_len = max(len(tensor) for tensor in out_tensors)

        self.train_inp, _, self.train_out, _ = train_test_split(inp_tensors, out_tensors, test_size=0.2)
        self.BUFFER_SIZE = len(self.train_inp)
        self.BATCH_SIZE = 64
        self.steps_per_epoch = self.BUFFER_SIZE // self.BATCH_SIZE
        self.embedding_dim = 256
        self.units = 1024


    def _preprocess(self, sentence):
        sentence = sentence.lower().strip()
        sentence = re.sub(r"([?.!,¿])", r" \1 ", sentence)
        sentence = re.sub(r'[" "]+', " ", sentence)
        remove_digits = str.maketrans('', '', digits)
        sentence = sentence.translate(remove_digits)
        sentence = re.sub('[२३०८१५७९४६|]', "", sentence)
        sentence = sentence.rstrip().strip()
        sentence = '<s> ' + sentence + ' <e>'
        return sentence


    def _tokenize(self, text):
        tokenizer = tf.keras.preprocessing.text.Tokenizer(filters='')
        tokenizer.fit_on_texts(text)
        text_tensors = tokenizer.texts_to_sequences(text)
        text_tensors = tf.keras.preprocessing.sequence.pad_sequences(text_tensors, padding='post')
        return text_tensors, tokenizer


    def train(self):
        inp_size = len(self._inp_tokenizer.word_index) + 1
        out_size = len(self._out_tokenizer.word_index) + 1
        self._encoder = self._Encoder(inp_size, self.embedding_dim, self.units, self.BATCH_SIZE)
        self._decoder = self._Decoder(out_size, self.embedding_dim, self.units, self.BATCH_SIZE)

        self._optimizer = Adam()
        self._loss_object = SCC(from_logits=True, reduction='none')

        self._checkpoint = tf.train.Checkpoint(optimizer=self._optimizer, encoder=self._encoder, decoder=self._decoder)

        if not os.path.exists('checkpoints'):
            dataset = tf.data.Dataset.from_tensor_slices((self.train_inp, self.train_out)).shuffle(self.BUFFER_SIZE)
            dataset = dataset.batch(self.BATCH_SIZE, drop_remainder=True)
            n_epochs = 12

            for i in range(n_epochs):
                start = time.time()
                print(f'EPOCH {i+1}')

                hidden = self._encoder.init_hidden()
                total_loss = 0

                for (batch, (inp, out)) in enumerate(dataset.take(self.steps_per_epoch)):
                    batch_loss = self._training_step(inp, out, hidden)
                    total_loss += batch_loss

                    if batch % 50 == 0: print(f'Completed {batch} batches, Loss : {batch_loss.numpy():.4f}')
                        
                if (i+1) % 2 == 0: self._checkpoint.save(file_prefix = os.path.join('./checkpoints', 'ckpt'))

                print(f'Epoch {i+1} completed, Loss : {(total_loss / self.steps_per_epoch):.4f}')
                print(f'Time taken : {time.time() - start:.4f} s')
                print('--'*20 + '\n')

        self._checkpoint.restore(tf.train.latest_checkpoint('./checkpoints'))  


    def _loss_function(self, real, prediction):
        mask = tf.math.logical_not(tf.math.equal(real, 0))
        loss = self._loss_object(real, prediction)
        mask = tf.cast(mask, dtype=loss.dtype)
        loss *= mask 
        return tf.reduce_mean(loss)


    @tf.function
    def _training_step(self, inp, out, hidden):
        loss = 0

        with tf.GradientTape() as tape:
            output, dec_hidden = self._encoder(inp, hidden)
            dec_input = tf.expand_dims([self._out_tokenizer.word_index['<s>']] * self.BATCH_SIZE, 1)

            for t in range(1, out.shape[1]):
                predictions, dec_hidden, _ = self._decoder(dec_input, dec_hidden, output)
                loss += self._loss_function(out[:, t], predictions)
                dec_input = tf.expand_dims(out[:, t], 1)
                
        batch_loss = (loss / int(out.shape[1]))
        variables = self._encoder.trainable_variables + self._decoder.trainable_variables
        gradients = tape.gradient(loss, variables)
        self._optimizer.apply_gradients(zip(gradients, variables))

        return batch_loss

    def translate(self, sentence):
        sentence = self._preprocess(sentence)

        inputs = [self._inp_tokenizer.word_index[i] for i in sentence.split()]
        inputs = pad_sequences([inputs], maxlen=self.max_inp_len, padding='post')
        inputs = tf.convert_to_tensor(inputs)

        result = ''

        hidden = [tf.zeros((1, self.units))]
        output, dec_hidden = self._encoder(inputs, hidden)
        dec_input = tf.expand_dims([self._out_tokenizer.word_index['<s>']], 0)

        for _ in range(self.max_out_len):
            pred, dec_hidden, _ = self._decoder(dec_input, dec_hidden, output)
            index = tf.argmax(pred[0]).numpy()
            result += self._out_tokenizer.index_word[index] + ' '
            if self._out_tokenizer.index_word[index] == '<e>': return result[:-4].strip()
            dec_input = tf.expand_dims([index], 0)
            
        return result[:-4].strip()


data = pd.read_csv('data.csv')

data = data[data['source'] == 'ted']
data = data[~pd.isnull(data['english_sentence'])]
data.drop_duplicates(inplace=True)

data = data.sample(n=35000, random_state=42)

inp_lang = [i for i in data['english_sentence']]
out_lang = [i for i in data['hindi_sentence']]

translator = Translator(inp_lang, out_lang)

translator.train()

hindi_sentence = translator.translate('What are you doing?')
print(hindi_sentence)

