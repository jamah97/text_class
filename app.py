

import streamlit as st




# Data Viz Pkg
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")
import seaborn as sns
st.set_option('deprecation.showPyplotGlobalUse', False)
from collections import Counter

import numpy as np
from numpy import argmax
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras import utils
from tensorflow.keras.layers import Dense, Activation, Dropout, Embedding
from sklearn import preprocessing
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OneHotEncoder


#ignore ValueError


def main():
    
    sms_data = pd.read_csv('spam.csv', delimiter=',',encoding='latin-1')
    x = sms_data['v2']
    y = sms_data['v1']

    train_x, test_x, train_y, test_y = train_test_split(x,y, test_size=.25, random_state=1)
    encoder = LabelEncoder()
    train_y = encoder.fit_transform(train_y)
    test_y = encoder.fit_transform(test_y)
    train_y = np.asarray(train_y).astype('float32').reshape((-1,1))
    test_y = np.asarray(test_y).astype('float32').reshape((-1,1))
    max_words = 10000
    max_len = 150
    embedding_dim = 50
    tok = Tokenizer(num_words=max_words, split=' ')
    tok.fit_on_texts(train_x)
    train_sequences = tok.texts_to_sequences(train_x)
    train_sequences = pad_sequences(train_sequences,maxlen=max_len)

    test_sequences = tok.texts_to_sequences(test_x)
    test_sequences = pad_sequences(test_sequences,maxlen=max_len)

    train_sequences = np.asarray(train_sequences).astype('float32')

    test_sequences = np.asarray(test_sequences).astype('float32')

    model = tf.keras.Sequential([
            tf.keras.layers.Embedding(max_words, embedding_dim, input_length=max_len),
            tf.keras.layers.Dense(24, activation='relu'),
            tf.keras.layers.Dropout(.2),
            tf.keras.layers.Dense(1, activation='sigmoid')
])

    model.compile(loss='binary_crossentropy',optimizer=tf.keras.optimizers.RMSprop(),metrics=['accuracy'])

    num_epochs = 5
    batch = 32

    history = model.fit(train_sequences, train_y, epochs=num_epochs, batch_size = batch, verbose=2, validation_data=(test_sequences, test_y))


    st.subheader("Text Classifier Predictor with Tensorflow")
    st.write("Datasource: https://www.kaggle.com/datasets/balaka18/email-spam-classification-dataset-csv")
    st.write("About dataset: containing related information of 5172 randomly picked email files and their respective labels for spam or not-spam classification.")
    st.write("In the 'Enter Text' section below, enter the header of the email you have received or the first couple of sentences in the email. Afterward, a prediction will be generated, whether it's spam or not. ")
    st.write('Data shortcoming: There is an imbalance of data. The ham to spam ratio is 6.45 to 1. Even though the accuracy of the model is 89.35% the uneven distribution may result in lots of regular emails being labeled as spam.')
    st.write(sms_data.v1.value_counts())

        #sms_data['v2'] = sms_data['v2'].apply(lambda x:len(str(x).split()))

        #st.write('Max length in main text for training:',sms_data['v2'].max())
        #st.write('Mean length in main text for training:',sms_data['v2'].mean())
        #st.write('Min length in main text for training:',sms_data['v2'].min())

        
        
    s = st.text_input('Enter Text').split()

    if st.button("Predict"):
        
        new_sequences = tok.texts_to_sequences(s)
        new_sequences1 = pad_sequences(new_sequences,maxlen=max_len)
        new_sequences1 = np.asarray(new_sequences1).astype('float32')
        st.write('List of words in text:',s)
        predicted = model.predict(new_sequences1)
        pred = np.max(predicted)
        if pred >= .50:
            st.write("Email is likly a spam", pred)
        else:
            st.write('Email is ham', pred)
        
        
        
if __name__ == '__main__':
    main()
