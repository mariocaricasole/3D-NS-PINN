# Import necessary libraries
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Reshape
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

#define a periodic layer using the cosine or sine as activation, using non-trainable parameters
class Periodic(keras.layers.Layer):
    def __init__(self, activation=tf.cos, freqs=16, input_dim=3):
        super().__init__()
        self.w = self.add_weight(
            shape=(input_dim,freqs**3), initializer=PeriodicWeights, trainable=False)
        self.activation = activation

    def call(self,inputs):
        return self.activation(2*np.pi*tf.matmul(inputs,self.w))

#define time-dependent amplitudes model
class AmplitudeModel(keras.Model):
    def __init__(self, freqs = 16, output_size = 4):
        super().__init__()
        self.freqs = freqs
        self.output_size = output_size

        #define the layers of the model
        self.dense_layer = Dense(output_size*freqs**3, activation='sigmoid')
        self.output_layer = Dense(output_size*freqs**3)
        self.reshape_layer = Reshape((output_size,freqs**3))

    def call(self,inputs):
        layer1 = self.dense_layer(inputs)
        output = self.output_layer(layer1)
        output = self.reshape_layer(output)

        return output

class NSModel(tf.keras.Model):
    def __init__(self, freqs=16, output_size=4):
        super().__init__()

        self.cos_periodic = Periodic(activation=tf.cos, freqs=freqs)
        self.sin_periodic = Periodic(activation=tf.sin, freqs=freqs)

        self.cos_amplitude = AmplitudeModel(freqs=freqs, output_size=output_size)
        self.sin_amplitude = AmplitudeModel(freqs=freqs, output_size=output_size)

    def call(self, inputs):
        # Split the input tensor into the first three variables and the last one
        periodic_inputs = inputs[:, :3]
        amplitude_input = inputs[:, 3:]

        # Compute outputs from periodic layers
        cos_output = self.cos_periodic(periodic_inputs)
        sin_output = self.sin_periodic(periodic_inputs)

        # Compute outputs from amplitude models
        cos_amplitude_output = self.cos_amplitude(amplitude_input)
        sin_amplitude_output = self.sin_amplitude(amplitude_input)

        # Element-wise multiplication and sum over frequencies
        combined_cos = tf.einsum('ijk,ik -> ij', cos_amplitude_output, cos_output)
        combined_sin = tf.einsum('ijk,ik -> ij', sin_amplitude_output, sin_output)

        # Sum the results along the last axis
        final_output = combined_cos + combined_sin

        return final_output
