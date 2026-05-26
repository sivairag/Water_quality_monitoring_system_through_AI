from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import pandas as pd
water_data = pd.read_csv('D:/project/water_quality/code/water_potability1.csv')

x = water_data.drop('Potability', axis =1).values 
y = water_data['Potability'].values

XX_train,XX_test,YY_train,YY_test=train_test_split(x,y,test_size=1/3,random_state=42)

scaler = StandardScaler()
XX_train_scaled = scaler.fit_transform(XX_train)
XX_test_scaled = scaler.transform(XX_test)
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM


lstm_classifier = Sequential([
    LSTM(64, input_shape=(XX_train_scaled.shape[1], 1)),
    Dense(1, activation='sigmoid')
])
lstm_classifier.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
lstm_classifier.fit(np.expand_dims(XX_train_scaled, axis=-1), YY_train, epochs=5, batch_size=8, validation_split=0.2)
lstm_pred_prob = lstm_classifier.predict(np.expand_dims(XX_test_scaled, axis=-1))
lstm_pred = (lstm_pred_prob > 0.5).astype(int).flatten()


lstm_accuracy = accuracy_score(YY_test, lstm_pred)


print(lstm_accuracy)
