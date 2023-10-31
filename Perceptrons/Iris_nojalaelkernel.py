#varianza = que tan dispersos estan
#sesgo = que tan cerca del centro esta

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from keras.models import Sequential #capas
from keras.layers import Dense #que las capas esten densamente conectadas
from sklearn.preprocessing import LabelEncoder #tranformar etiqueta en numero
from sklearn.model_selection import train_test_split

#Largo del sepalo / Ancho del sepalo / Largo del petalo / Ancho del petalo /Especie
#cargar los datos desde el archivo iris.data
data = pd.read_csv(r"iris.csv", header=None)

training_data = data.drop([4], axis=1) #solo columnas  0,1,2,3
target_data = data.drop([0,1,2,3], axis=1) #solo la columna 4

#convertir etiquetas de texto a números
label_encoder = LabelEncoder()
target_data = label_encoder.fit_transform(target_data)

#separar los datos 70% entrenar, 30% prueba
validation_size = 0.30
seed = 7
X_train, X_test, y_train, y_test = train_test_split(training_data, target_data, test_size=validation_size, random_state=seed)

#crear el modelo
model=Sequential()
#3 neuronas = setosa, versicolor, virginica
#4 var = largo y ancho del sepalo y el petalo
model.add(Dense(3, input_dim = 4, activation = 'tanh'))
#model.add(Dense(4, activation='sigmoid')) #estos tres descomentar y ver como afecta al final
#model.add(Dense(2, activation='sigmoid'))
#model.add(Dense(1, activation='sigmoid'))
model.compile(loss='mean_squared_error',
              optimizer='adam', metrics=['accuracy'])

#entrenar el modelo
#history=model.fit(X_train, y_train, epochs = 2000, batch_size = 5, validation_data = (X_test, y_test), verbose = 1)
history = model.fit(X_train, y_train, epochs = 100, batch_size = 5, validation_data = (X_test, y_test), verbose = 1)

#evaluar el modelo
scores = model.evaluate(training_data, target_data)
loss, accuracy = model.evaluate(X_test, y_test, verbose=0)

print('Accuracy:', accuracy)
#print("\n%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))
#print(model.predict(training_data).round())

model.summary()
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Accuracy del modelo')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Entrenamiento', 'Test'], loc='upper left')
plt.show()