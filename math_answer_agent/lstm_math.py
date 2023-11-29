import os, sys, time
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import pandas as pd

from numpy.random import seed
seed(4)

import seaborn as sns
# custom_params = {"axes.spines.right": False, "axes.spines.top": False}
# custom_params = {"axes.spines.top": False}
# sns.set_theme(style="ticks", rc=custom_params)
sns.set_theme(style="ticks")
# sns.set_style("darkgrid")

from sklearn.metrics import confusion_matrix
import tensorflow as tf 
tf.random.set_seed(4)
from tensorflow import keras
from tensorflow.keras import layers

class CharacterTable:
    def __init__(self, chars):
        self.chars = sorted(set(chars))
        self.char_indices = dict((c, i) for i, c in enumerate(self.chars))
        self.indices_char = dict((i, c) for i, c in enumerate(self.chars))

    def encode(self, C, num_rows):
        x = np.zeros((num_rows, len(self.chars)))
        for i, c in enumerate(C):
            x[i, self.char_indices[c]] = 1
        return x

    def decode(self, x, calc_argmax=True):
        if calc_argmax:
            x = x.argmax(axis=-1)
        return "".join(self.indices_char[x] for x in x)

def data_prepare(dataset_path, MAXLEN, chars, shuffle = False, split_percent = 10):
    train_set = pd.read_csv(dataset_path)
    questions, expected = [], []
    for i in range(len(train_set)):
        sample = train_set.loc[i]
        num_1, num_2, num_3, truth = int(sample['num_1']), int(sample['num_2']), int(sample['num_3']), int(sample['answer'])
        questions.append(str(num_1) + '≡' + str(num_2) + '(mod' + str(num_3) + ')')
        expected.append(str(truth))

    ctable = CharacterTable(chars)
    x, y = np.zeros((len(questions), MAXLEN, len(chars)), dtype=np.bool), np.zeros((len(questions), 1, len(chars)), dtype=np.bool)
    for i, sentence in enumerate(questions):
        x[i] = ctable.encode(sentence, MAXLEN)
    for i, sentence in enumerate(expected):
        y[i] = ctable.encode(sentence, 1)

    if shuffle == True:
        indices = np.arange(len(y))
        np.random.shuffle(indices)
        x, y = x[indices], y[indices]
    if split_percent != 0:
        split_at = len(x) - len(x) // split_percent
        (x_train, x_val) = x[:split_at], x[split_at:]
        (y_train, y_val) = y[:split_at], y[split_at:]
        print(f"Train Data: x_train.shape: {x_train.shape}, y_train.shape: {y_train.shape}")
        print(f"Valid Data: x_val.shape: {x_val.shape}, y_val.shape: {y_val.shape}")
        return x_train, y_train, x_val, y_val
    else:
        print(f"Data: x.shape: {x.shape}, y.shape: {y.shape}")
        return x, y


def make_model(MAXLEN, chars, neuron_num):
    num_layers = 1  
    model = keras.Sequential()
    model.add(layers.LSTM(neuron_num, input_shape=(MAXLEN, len(chars))))
    model.add(layers.RepeatVector(1))
    for _ in range(num_layers):
        model.add(layers.LSTM(neuron_num, return_sequences=True))
    model.add(layers.Dense(len(chars), activation="softmax"))
    model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
    model.summary()
    model_exp = keras.Model(inputs = model.input, outputs = model.get_layer('dense').output)
    tf.keras.utils.plot_model(model, to_file='model.png', show_shapes=True)
    return model, model_exp

def model_train_val(model, model_exp, x_train, y_train, x_val, y_val, chars, epoch_num, batch_size, history_path):
    ctable = CharacterTable(chars)
    history_table = pd.DataFrame(columns=['loss', 'accuracy', 'val_loss', 'val_accuracy'])
    for epoch in range(1, epoch_num+1):
        print()
        print("Iteration", epoch)
        history = model.fit(x_train, y_train, batch_size=batch_size, epochs=1, validation_data=(x_val, y_val))
        new_history = pd.DataFrame([[history.history['loss'][0], history.history['accuracy'][0], history.history['val_loss'][0], history.history['val_accuracy'][0]]], columns=['loss', 'accuracy', 'val_loss', 'val_accuracy'])
        history_table = pd.concat([history_table, new_history],ignore_index=True)
        model.save(MODEL_PATH)
    
        for i in range(10):
            ind = np.random.randint(0, len(x_val))
            rowx, rowy = x_val[np.array([ind])], y_val[np.array([ind])]
            prediction = model.predict(rowx)
            preds = np.argmax(prediction, axis=-1)
            q = ctable.decode(rowx[0])
            correct = ctable.decode(rowy[0])
            guess = ctable.decode(preds[0], calc_argmax=False)
            print("Q", q, end=" ")
            print("T", correct, end=" ")
            if correct == guess:
                print("T: " + guess)
            else:
                print("F: " + guess)   
    history_table.to_csv(history_path,index=False)       

def model_test(model_path, test_dataset_path, MAXLEN, chars, output_file = None):
    model = keras.models.load_model(model_path)
    x_test, y_test = data_prepare(test_dataset_path, MAXLEN, chars, shuffle = False, split_percent = 0)
    correct_counter = 0
    ctable = CharacterTable(chars)
    pred_results = []
    for i in range(len(y_test)):
        rowx, rowy = x_test[np.array([i])], y_test[np.array([i])]
        prediction = model.predict(rowx)
        preds = np.argmax(prediction, axis=-1)
        q = ctable.decode(rowx[0])
        correct = ctable.decode(rowy[0])
        guess = ctable.decode(preds[0], calc_argmax=False)
        pred_results.append(int(guess))
        print("Q", q, end=" ")
        print("T", correct, end=" ")
        if correct == guess:
            correct_counter += 1
            print("T: " + guess)
        else:
            print("F: " + guess)  
    print(f'accuracy: {correct_counter/len(y_test)}')
    if output_file != None:
        test_table = pd.read_csv(test_dataset_path)
        pred_results = np.array(pred_results).reshape((len(pred_results),1))
        test_table['pred'] = pred_results
        test_table.to_csv(output_file, index = False)

def plot_confusion_matrix(output_file):
    output_table = pd.read_csv(output_file)
    y_true, y_pred = np.array(output_table['answer']), np.array(output_table['pred'])
    print(f'accuracy: {np.sum(y_true==y_pred)/len(y_pred)}')
    matrix = confusion_matrix(y_true, y_pred)
    ax = sns.heatmap(matrix, annot=True, fmt="d", cmap="YlGnBu") # , cmap="YlGnBu"
    plt.show()

def plot_loss_accuracy(history_path):
    history = pd.read_csv(history_path)
    history['epoch'] = np.arange(0,len(history)).reshape((len(history),1))
    # plt.subplot(121)
    ax = sns.lineplot(x='epoch', y='loss', data = history, label = 'loss', palette="flare")
    sns.lineplot(x='epoch', y='val_loss', data = history, label = 'val loss', palette="flare") 
    # plt.subplot(122)
    ax_2 = ax.axes.twinx()
    sns.lineplot(x='epoch', y='accuracy', data = history, label = 'accuracy', palette="flare", ax=ax_2)
    sns.lineplot(x='epoch', y='val_accuracy', data = history, label = 'val accuracy', palette="flare", ax=ax_2) 
    plt.show()

neuron_num = 256
epoch_num = 100
batch_size = 8
MAXLEN, chars = 11, "0123456789≡(mod) "
train_dataset = 'train_numeric.csv'
test_dataset = 'test_numeric.csv'
MODEL_PATH = 'math_answer_model_' + str(neuron_num)
history_path = 'model_history_'+str(neuron_num)+'.csv'
predict_path = 'test_numeric_out_'+str(neuron_num)+'.csv'

# model, model_exp = make_model(MAXLEN, chars, neuron_num)
# x_train, y_train, x_val, y_val = data_prepare(train_dataset, MAXLEN, chars, shuffle = True)
# model_train_val(model, model_exp, x_train, y_train, x_val, y_val, chars, epoch_num, batch_size, history_path)

# model = keras.models.load_model(MODEL_PATH)
# model_exp = keras.Model(inputs = model.input, outputs = model.get_layer('lstm_1').output)

# model_test(MODEL_PATH, test_dataset, MAXLEN, chars, output_file = predict_path)

# plot_confusion_matrix(predict_path)
# plot_loss_accuracy(history_path)