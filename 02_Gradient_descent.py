import tensorflow as tf
tf.set_random.set_seed(0)

x_daat = [1.0, 2.0, 3.0, 4.]
y_daat = [1.0, 2.0, 3.0, 4.]

W = tf.Variable(tf.random_normal([1], -100., 100.))

for step in range(300):
    hypothesis = W * x_daat
    cost = tf.reduce_mean(tf.square(hypothesis - y_daat))

    aplha = 0.01
    gradient = tf.reduce_mean(tf.multiply(tf.multiply(W, x_daat) - y_daat, x_daat))
    descnet = W - tf.multiply(aplha, gradient)
    W.assign(descnet)

    if step % 10 == 0:
        print("Step: {:5}\tW: {:10.4f}\tcost: {:10.6f}".format(step, cost.numpy(), W.numpy()[0]))