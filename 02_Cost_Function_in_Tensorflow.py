import tensorflow as tf
import numpy as np

X = np.array([1, 2, 3])
Y = np.array([1, 2, 3])

def cost_func(W, X, Y):
    hypothesis = W * X
    return tf.reduce_mean(tf.square(hypothesis - Y))

W_values = tf.linspace(-3.0, 5.0, num=15)
cost_values = []

print("\n")
for feed_W in W_values:
    curr_cost = cost_func(feed_W, X, Y)
    cost_values.append(curr_cost)
    print("W: {:6.3f}, cost: {:10.5f}".format(feed_W, curr_cost))