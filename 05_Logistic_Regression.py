# # 기존 코드
# import tensorflow as tf
# import tensorflow.contrib.eager as tfe
# import numpy as np
# tf.enable_eager_execution()

# x_train = [
#     [1., 2.],
#     [2., 3.],
#     [3., 1.],
#     [4., 3.],
#     [5., 3.],
#     [6., 2.]
# ]
# y_train = [
#     [0.],
#     [0.],
#     [0.],
#     [1.],
#     [1.],
#     [1.]
# ]
# x_test = [[5., 2.]]
# y_test = [[1.]]

# x_test = np.array([[5, 2]], dtype=np.float32)
# y_test = np.array([[1]], dtype=np.float32)

# dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train)).batch(len(x_train))
# W = tf.Variable(tf.zeros([2, 1]), name='weight')
# b = tf.Variable(tf.zeros([1]), name='bias')

# def logistic_regression(features):
#     hypothesis = tf.div(1., 1. + tf.exp(tf.matmul(features, W) + b))
#     return hypothesis

# def loss_fn(features, labels):
#     hypothesis = logistic_regression(features)
#     cost = -tf.reduce_mean(labels * tf.log(hypothesis) + (1 - labels) * tf.log(1 - hypothesis))
#     return cost

# def grad(hypothesis, features, labels):
#     with tf.GradientTape() as tape:
#         loss_value = loss_fn(hypothesis, labels)
#     return tape.gradient(loss_value, [W, b])

# optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01)

# EPOCHS = 3000

# for step in range(EPOCHS + 1):
#     for features, labels in tfe.Iterator(dataset):
#         grads = grad(logistic_regression(features), features, labels)
#         optimizer.apply_gradients(grads_and_vars=zip(grads, [W, b]))
#         if step % 100 == 0:
#             print("Iter: {}, Loss: {:.4f}".format(step, loss_fn(logistic_regression(features), labels)))

# def accuracy_fn(hypothesis, labels):
#     predicted = tf.cast(hypothesis > 0.5, dtype=tf.float32)
#     accuracy = tf.reduce_mean(tf.cast(tf.equal(predicted, labels), dtype=tf.int32))
#     return accuracy

# test_acc = accuracy_fn(logistic_regression(x_test), y_test)


# # 새로 수정한 코드1
import tensorflow as tf
import numpy as np

x_train = [
    [1., 2.],
    [2., 3.],
    [3., 1.],
    [4., 3.],
    [5., 3.],
    [6., 2.]
]
y_train = [
    [0.],
    [0.],
    [0.],
    [1.],
    [1.],
    [1.]
]
x_test = [[5., 2.]]
y_test = [[1.]]

dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train)).batch(len(x_train))
W = tf.Variable(tf.zeros([2, 1]), name='weight')
b = tf.Variable(tf.zeros([1]), name='bias')

def logistic_regression(features):
    z = tf.matmul(features, W) + b
    hypothesis = 1. / (1. + tf.exp(-z))
    return hypothesis

def loss_fn(features, labels):
    p = logistic_regression(features)
    return -tf.reduce_mean(labels*tf.math.log(p) + (1-labels)*tf.math.log(1-p))

optimizer = tf.keras.optimizers.SGD(learning_rate=0.01)

EPOCHS = 3000

for step in range(EPOCHS + 1):
    for features, labels in dataset:
        with tf.GradientTape() as tape:
            loss = loss_fn(features, labels)
        grads = tape.gradient(loss, [W, b])
        optimizer.apply_gradients(zip(grads, [W, b]))
        if step % 100 == 0:
            print(f"Iter: {step}, Loss: {loss.numpy():.4f}")

def accuracy_fn(features, labels):
    p = logistic_regression(features)
    pred = tf.cast(p > 0.5, tf.float32)
    return tf.reduce_mean(tf.cast(tf.equal(pred, labels), tf.float32))

test_acc = accuracy_fn(x_test, y_test)
print("test_acc:", float(test_acc))


# # 새로 수정한 코드2
# import tensorflow as tf
# import numpy as np

# x_train = np.array([
#     [1, 2],
#     [2, 3],
#     [3, 1],
#     [4, 3],
#     [5, 3],
#     [6, 2]], dtype=np.float32)
# y_train = np.array([
#     [0],
#     [0],
#     [0],
#     [1],
#     [1],
#     [1]], dtype=np.float32)

# x_test = np.array([[5, 2]], dtype=np.float32)
# y_test = np.array([[1]], dtype=np.float32)

# # tf.data.Dataset 파이프라인을 이용하여 값을 입력
# # from_tensor_slices 클래스 매서드를 사용하면 리스트, 넘파이, 텐서플로 자료형에서 데이터셋을 만들 수 있음
# dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train)).batch(len(x_train))
# W = tf.Variable(tf.zeros([2, 1]), name='weight')
# b = tf.Variable(tf.zeros([1]), name='bias')

# # 원소의 자료구조 반환
# dataset.element_spec

# def logistic_regression(features):
#     hypothesis = tf.sigmoid(tf.matmul(features, W) + b)
#     return hypothesis


# def loss_fn(features, labels):
#     hypothesis = logistic_regression(features)
#     cost = -tf.reduce_mean(labels * tf.math.log(hypothesis) + (1 - labels) * tf.math.log(1 - hypothesis))
#     return cost

# def grad(hypothesis, features, labels):
#     with tf.GradientTape() as tape:
#         loss_value = loss_fn(features, labels)
#     return tape.gradient(loss_value, [W,b])

# optimizer = tf.keras.optimizers.SGD(learning_rate=0.01)

# EPOCHS = 3000

# for step in range(EPOCHS + 1):
#     for features, labels in iter(dataset):
#         hypothesis = logistic_regression(features)
#         grads = grad(hypothesis, features, labels)
#         optimizer.apply_gradients(grads_and_vars=zip(grads, [W,b]))
#         if step % 300 == 0:
#             print("Iter: {}, Loss: {:.4f}".format(step, loss_fn(features, labels)))
#             # print(hypothesis)
            
# def accuracy_fn(hypothesis, labels):
#     predicted = tf.cast(hypothesis > 0.5, dtype=tf.float32)
#     accuracy = tf.reduce_mean(tf.cast(tf.equal(predicted, labels), dtype=tf.int32))
#     return accuracy

# test_acc = accuracy_fn(logistic_regression(x_test), y_test)
# print('Accuracy: {}%'.format(test_acc * 100))