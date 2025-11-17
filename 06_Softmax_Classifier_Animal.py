import tensorflow as tf
import numpy as np
tf.get_logger().setLevel('ERROR')

# 1) Sample Dataset 로드 (로컬 파일 사용)
xy = np.loadtxt('data-04-zoo.csv', delimiter=',', dtype=np.float32)
x_data = xy[:, 0:-1]      # (N, 16)
y_data = xy[:, [-1]]      # (N, 1)

nb_classes = 7  # 0 ~ 6

# (N, 1) 정수 라벨 → (N,) → one_hot (N, 7)
Y_one_hot = tf.one_hot(tf.squeeze(tf.cast(y_data, tf.int32), axis=1), depth=nb_classes)
Y_one_hot = tf.reshape(Y_one_hot, [-1, nb_classes])

# 2) 파라미터 정의
W = tf.Variable(tf.random.normal([16, nb_classes]), name='weight')
b = tf.Variable(tf.random.normal([nb_classes]), name='bias')
variables = [W, b]

# 3) 모델/손실/평가 함수
def logit_fn(X):
    return tf.matmul(X, W) + b

def hypothesis(X):
    return tf.nn.softmax(logit_fn(X))

def cost_fn(X, Y):
    logits = logit_fn(X)
    cost_i = tf.nn.softmax_cross_entropy_with_logits(labels=Y, logits=logits)
    cost = tf.reduce_mean(cost_i)
    return cost

def grad_fn(X, Y):
    with tf.GradientTape() as tape:
        loss = cost_fn(X, Y)
    grads = tape.gradient(loss, variables)
    return grads  # [dW, db]

def prediction(X, Y):
    pred = tf.argmax(hypothesis(X), axis=1)
    correct_prediction = tf.equal(pred, tf.argmax(Y, axis=1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    return accuracy

# 4) 학습 루프
optimizer = tf.keras.optimizers.SGD(learning_rate=0.1)

def fit(X, Y, epochs=500, verbose=50):
    for i in range(epochs):
        grads = grad_fn(X, Y)
        optimizer.apply_gradients(zip(grads, variables))
        if (i == 0) | ((i + 1) % verbose == 0):
            acc  = prediction(X, Y).numpy()
            loss = tf.reduce_sum(cost_fn(X, Y)).numpy()
            print('Loss & Acc at {} epoch {}, {}'.format(i + 1, loss, acc))

# 5) 실행
fit(x_data, Y_one_hot)
