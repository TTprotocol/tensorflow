# import os
# os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"  # 0: INFO, 1: WARNING, 2: ERROR, 3: FATAL
# import tensorflow as tf
# tf.get_logger().setLevel('ERROR')
# import numpy as np

# tf.random.set_seed(777)

# # --- XOR toy data ---
# x_data = np.array([[0, 0],
#                    [0, 1],
#                    [1, 0],
#                    [1, 1]], dtype=np.float32)
# y_data = np.array([[0],
#                    [1],
#                    [1],
#                    [0]], dtype=np.float32)

# # Dataset
# dataset = tf.data.Dataset.from_tensor_slices((x_data, y_data)).batch(len(x_data))


# def preprocess_data(features, labels):
#     features = tf.cast(features, tf.float32)
#     labels = tf.cast(labels, tf.float32)
#     return features, labels


# # --- Weights / Biases ---
# W1 = tf.Variable(tf.random.normal([2, 10]), name="weight1")
# b1 = tf.Variable(tf.random.normal([10]), name="bias1")

# W2 = tf.Variable(tf.random.normal([10, 10]), name="weight2")
# b2 = tf.Variable(tf.random.normal([10]), name="bias2")

# W3 = tf.Variable(tf.random.normal([10, 10]), name="weight3")
# b3 = tf.Variable(tf.random.normal([10]), name="bias3")

# W4 = tf.Variable(tf.random.normal([10, 1]), name="weight4")
# b4 = tf.Variable(tf.random.normal([1]), name="bias4")


# def neural_net(features):
#     layer1 = tf.sigmoid(tf.matmul(features, W1) + b1)
#     layer2 = tf.sigmoid(tf.matmul(layer1, W2) + b2)
#     layer3 = tf.sigmoid(tf.matmul(layer2, W3) + b3)
#     hypothesis = tf.sigmoid(tf.matmul(layer3, W4) + b4)
#     return layer1, layer2, layer3, hypothesis


# def loss_fn(hypothesis, labels):
#     # Binary cross-entropy
#     cost = -tf.reduce_mean(
#         labels * tf.math.log(hypothesis + 1e-7)
#         + (1.0 - labels) * tf.math.log(1.0 - hypothesis + 1e-7)
#     )
#     return cost


# optimizer = tf.keras.optimizers.SGD(learning_rate=0.1)

# log_path = "./logs/xor_eager_tf2"
# writer = tf.summary.create_file_writer(log_path)


# def accuracy_fn(hypothesis, labels):
#     predicted = tf.cast(hypothesis > 0.5, dtype=tf.float32)
#     acc = tf.reduce_mean(tf.cast(tf.equal(predicted, labels), dtype=tf.float32))
#     return acc


# @tf.function
# def train_step(features, labels, step):
#     with tf.GradientTape() as tape:
#         layer1, layer2, layer3, hypothesis = neural_net(features)
#         loss_value = loss_fn(hypothesis, labels)

#     grads = tape.gradient(
#         loss_value,
#         [W1, W2, W3, W4, b1, b2, b3, b4],
#     )
#     optimizer.apply_gradients(
#         zip(grads, [W1, W2, W3, W4, b1, b2, b3, b4])
#     )

#     # TensorBoard summary
#     with writer.as_default():
#         tf.summary.scalar("loss", loss_value, step=step)
#         tf.summary.histogram("weights1", W1, step=step)
#         tf.summary.histogram("biases1", b1, step=step)
#         tf.summary.histogram("layer1", layer1, step=step)
#         tf.summary.histogram("weights2", W2, step=step)
#         tf.summary.histogram("biases2", b2, step=step)
#         tf.summary.histogram("layer2", layer2, step=step)
#         tf.summary.histogram("weights3", W3, step=step)
#         tf.summary.histogram("biases3", b3, step=step)
#         tf.summary.histogram("layer3", layer3, step=step)
#         tf.summary.histogram("weights4", W4, step=step)
#         tf.summary.histogram("biases4", b4, step=step)
#         tf.summary.histogram("hypothesis", hypothesis, step=step)

#     return loss_value

# EPOCHS = 3000
# step = 0

# for epoch in range(EPOCHS):
#     for features, labels in dataset:
#         features, labels = preprocess_data(features, labels)
#         step += 1
#         loss_value = train_step(features, labels, step)

#     if epoch % 500 == 0:
#         print(f"Epoch: {epoch}, Loss: {loss_value.numpy():.4f}")

# # 평가
# x_test = tf.cast(x_data, tf.float32)
# y_test = tf.cast(y_data, tf.float32)
# _, _, _, hypothesis = neural_net(x_test)
# test_acc = accuracy_fn(hypothesis, y_test)
# print(f"Testset Accuracy: {test_acc.numpy():.4f}")

# print("Predictions:\n", tf.cast(hypothesis > 0.5, tf.int32).numpy())

# # TensorBoard 실행 방법 (터미널):
# # tensorboard --logdir=./logs/xor_eager_tf2
# # 브라우저에서 http://127.0.0.1:6006 접속

import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"  # 0: INFO, 1: WARNING, 2: ERROR, 3: FATAL
import tensorflow as tf
tf.get_logger().setLevel('ERROR')
import numpy as np

tf.random.set_seed(777)

# --- XOR toy data ---
x_data = np.array([[0, 0],
                   [0, 1],
                   [1, 0],
                   [1, 1]], dtype=np.float32)
y_data = np.array([[0],
                   [1],
                   [1],
                   [0]], dtype=np.float32)

# Dataset
dataset = tf.data.Dataset.from_tensor_slices((x_data, y_data)).batch(len(x_data))


def preprocess_data(features, labels):
    features = tf.cast(features, tf.float32)
    labels = tf.cast(labels, tf.float32)
    return features, labels


# --- Weights / Biases ---
W1 = tf.Variable(tf.random.normal([2, 10]), name="weight1")
b1 = tf.Variable(tf.random.normal([10]), name="bias1")

W2 = tf.Variable(tf.random.normal([10, 10]), name="weight2")
b2 = tf.Variable(tf.random.normal([10]), name="bias2")

W3 = tf.Variable(tf.random.normal([10, 10]), name="weight3")
b3 = tf.Variable(tf.random.normal([10]), name="bias3")

W4 = tf.Variable(tf.random.normal([10, 1]), name="weight4")
b4 = tf.Variable(tf.random.normal([1]), name="bias4")


def neural_net(features):
    layer1 = tf.sigmoid(tf.matmul(features, W1) + b1)
    layer2 = tf.sigmoid(tf.matmul(layer1, W2) + b2)
    layer3 = tf.sigmoid(tf.matmul(layer2, W3) + b3)
    hypothesis = tf.sigmoid(tf.matmul(layer3, W4) + b4)
    return layer1, layer2, layer3, hypothesis


def loss_fn(hypothesis, labels):
    # Binary cross-entropy
    cost = -tf.reduce_mean(
        labels * tf.math.log(hypothesis + 1e-7)
        + (1.0 - labels) * tf.math.log(1.0 - hypothesis + 1e-7)
    )
    return cost


optimizer = tf.keras.optimizers.SGD(learning_rate=0.1)

log_path = "./logs/xor_eager_tf2"
writer = tf.summary.create_file_writer(log_path)


def accuracy_fn(hypothesis, labels):
    predicted = tf.cast(hypothesis > 0.5, dtype=tf.float32)
    acc = tf.reduce_mean(tf.cast(tf.equal(predicted, labels), dtype=tf.float32))
    return acc


# 변경: train_step에서 step 인자 제거, summary도 제거하고 "순수 학습"만 하도록 변경
@tf.function
def train_step(features, labels):
    with tf.GradientTape() as tape:
        layer1, layer2, layer3, hypothesis = neural_net(features)
        loss_value = loss_fn(hypothesis, labels)

    grads = tape.gradient(
        loss_value,
        [W1, W2, W3, W4, b1, b2, b3, b4],
    )
    optimizer.apply_gradients(
        zip(grads, [W1, W2, W3, W4, b1, b2, b3, b4])
    )

    return loss_value, layer1, layer2, layer3, hypothesis


EPOCHS = 3000
global_step = 0  # 변경: step 대신 global_step 사용

for epoch in range(EPOCHS):
    for features, labels in dataset:
        features, labels = preprocess_data(features, labels)
        loss_value, layer1, layer2, layer3, hypothesis = train_step(features, labels)
        global_step += 1

    # epoch마다 로그 출력
    if epoch % 500 == 0:
        print(f"Epoch: {epoch}, Loss: {loss_value.numpy():.4f}")

    # 변경: TensorBoard summary는 epoch 기준으로, 간격을 두고 기록
    if epoch % 100 == 0:
        with writer.as_default():
            tf.summary.scalar("loss", loss_value, step=global_step)
            tf.summary.histogram("weights1", W1, step=global_step)
            tf.summary.histogram("biases1", b1, step=global_step)
            tf.summary.histogram("layer1", layer1, step=global_step)
            tf.summary.histogram("weights2", W2, step=global_step)
            tf.summary.histogram("biases2", b2, step=global_step)
            tf.summary.histogram("layer2", layer2, step=global_step)
            tf.summary.histogram("weights3", W3, step=global_step)
            tf.summary.histogram("biases3", b3, step=global_step)
            tf.summary.histogram("layer3", layer3, step=global_step)
            tf.summary.histogram("weights4", W4, step=global_step)
            tf.summary.histogram("biases4", b4, step=global_step)
            tf.summary.histogram("hypothesis", hypothesis, step=global_step)

# 평가
x_test = tf.cast(x_data, tf.float32)
y_test = tf.cast(y_data, tf.float32)
_, _, _, hypothesis = neural_net(x_test)
test_acc = accuracy_fn(hypothesis, y_test)
print(f"Testset Accuracy: {test_acc.numpy():.4f}")

print("Predictions:\n", tf.cast(hypothesis > 0.5, tf.int32).numpy())

# TensorBoard 실행 방법 (터미널):
# tensorboard --logdir=./logs/xor_eager_tf2
# 브라우저에서 http://127.0.0.1:6006 접속
