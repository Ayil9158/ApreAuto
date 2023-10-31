import numpy as np
from keras.models import Sequential
from keras.layers import Dense #, Activation  # Para capas densas (fully connected) y funciones de activación
import matplotlib.pyplot as plt

# Cargamos las 4 combinaciones de las compuertas XOR, AND y OR
training_data=np.array([[0,0], [0,1], [1,0], [1,1]], "float32")

# Resultados que se obtienen en el mismo orden, descomentar el que se quiera usar
target_data=np.array([[0],[1],[1],[0]], "float32")
#target_data=np.array([[0],[0],[0],[1]], "float32")
#target_data=np.array([[0],[1],[1],[1]], "float32")

 # XOR -> 2500e=50%, 2000e=50%, 1500e=75%, 1000e=75%
 #AND -> 2000e=100%, 1500e=100%, 1000e=75%
 # OR -> 2000e=100%, 1500e=100%, 1000e=75%
model=Sequential()
model.add(Dense(2,input_dim=2, activation='tanh'))
model.add(Dense(4, activation='sigmoid'))
model.add(Dense(2, activation='sigmoid'))
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='mean_squared_error',
              optimizer='adam', metrics=['accuracy'])

history=model.fit(training_data, target_data, epochs=1500)

#evaluamos el modelo
scores=model.evaluate(training_data, target_data)

print("\n%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))
print(model.predict(training_data).round())

model.summary()
plt.plot(history.history['accuracy'])
plt.plot(history.history['loss'])
plt.title('Accuracy del modelo')
plt.ylabel('Accuracy')
plt.xlabel('Epocas')
plt.legend(['Entrenamiento','Test'], loc='upper left')
plt.show()