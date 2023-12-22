# Import necessary libraries
import numpy as np
from keras.models import Sequential
from keras.layers import Dense
import keras
import tensorflow as tf
import copy

#define custom weight initializer, essentially the 3D Fourier series frequency components
class PeriodicWeights(keras.initializers.Initializer):
    #define call for periodic integer weights
    def __call__(self, shape, dtype=tf.int8):
        freqs = int(np.cbrt(shape[1]))
        lin = tf.linspace(0.,freqs-1, freqs)
        kX, kY, kZ = tf.meshgrid(lin,lin,lin)
        kVec = tf.stack([kX,kY,kZ],axis=0)
        tensor = tf.reshape(kVec, [shape[0], shape[1]])
        return tensor

#define a periodic layer using the cosine as activation and a fixed phase, defined with trainable parameters
class Periodic(keras.layers.Layer):
    def __init__(self, freqs=16, input_dim=4):
        super().__init__()
        self.w = self.add_weight(
            shape=(input_dim,freqs**3), initializer=PeriodicWeights, trainable=False)
        self.b = self.add_weight(
            shape=(freqs**3,), initializer="zeros", trainable=True)

    def call(self,inputs):
        return tf.cos(2*np.pi*tf.matmul(inputs,self.w) + self.b)

#define fixed amplitudes model
model = Sequential(
    [
        Periodic(20,3),
        Dense(1)
    ])

model.build((None,3))
model.compile(optimizer='adamax', loss='mse')

#define initial condition
def f(xVec):
    x = xVec[:,0]
    y = xVec[:,1]
    z = xVec[:,2]

    return tf.cos(2*np.pi*z) + tf.sin(2*np.pi*(3*z + 4*x))

#define grid for spatial coordinate
iterations = 1000
for i in range(iterations):
    points = tf.random.uniform([100,3],0.0,1.0)
    model.fit(points,f(points))
