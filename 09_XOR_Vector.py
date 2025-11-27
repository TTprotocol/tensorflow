import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"  # 0: INFO, 1: WARNING, 2: ERROR, 3: FATAL
import tensorflow as tf
tf.get_logger().setLevel('ERROR')
tf.random.set_seed(777)

# XOR 데이터
x_data = tf.constant([[0., 0.],
                      [0., 1.],
                      [1., 0.],
                      [1., 1.]], dtype=tf.float32)

y_data = tf.constant([[0.],
                      [1.],
                      [1.],
                      [0.]], dtype=tf.float32)

dataset = tf.data.Dataset.from_tensor_slices((x_data, y_data)).batch(len(x_data))


def preprocess_data(features, labels):
    return tf.cast(features, tf.float32), tf.cast(labels, tf.float32)


# W1: [2, 2], b1: [2]  → hidden layer (뉴런 2개)
W1 = tf.Variable(tf.random.normal([2, 2]), name="weight1")
b1 = tf.Variable(tf.random.normal([2]), name="bias1")

# W2: [2, 1], b2: [1]  → output layer
W2 = tf.Variable(tf.random.normal([2, 1]), name="weight2")
b2 = tf.Variable(tf.random.normal([1]), name="bias2")


def neural_net(features):
    layer = tf.sigmoid(tf.matmul(features, W1) + b1)   # hidden layer
    hypothesis = tf.sigmoid(tf.matmul(layer, W2) + b2) # output layer
    return hypothesis


def loss_fn(hypothesis, labels):
    cost = -tf.reduce_mean(
        labels * tf.math.log(hypothesis + 1e-7)
        + (1.0 - labels) * tf.math.log(1.0 - hypothesis + 1e-7)
    )
    return cost


optimizer = tf.keras.optimizers.SGD(learning_rate=0.1)


@tf.function
def train_step(features, labels):
    with tf.GradientTape() as tape:
        hypothesis = neural_net(features)
        loss_value = loss_fn(hypothesis, labels)
    grads = tape.gradient(loss_value, [W1, W2, b1, b2])
    optimizer.apply_gradients(zip(grads, [W1, W2, b1, b2]))
    return loss_value


def accuracy_fn(hypothesis, labels):
    predicted = tf.cast(hypothesis > 0.5, dtype=tf.float32)
    accuracy = tf.reduce_mean(tf.cast(tf.equal(predicted, labels), dtype=tf.float32))
    return accuracy


EPOCHS = 10000

for step in range(EPOCHS):
    for features, labels in dataset:
        features, labels = preprocess_data(features, labels)
        loss_value = train_step(features, labels)
    if step % 1000 == 0:
        print(f"Iter: {step}, Loss: {loss_value.numpy():.4f}")

x_test, y_test = preprocess_data(x_data, y_data)
test_hypothesis = neural_net(x_test)
test_acc = accuracy_fn(test_hypothesis, y_test)
print(f"Testset Accuracy: {test_acc.numpy():.4f}")
print("Predictions:\n", tf.cast(test_hypothesis > 0.5, tf.int32).numpy())