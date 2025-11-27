import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"  # 0: INFO, 1: WARNING, 2: ERROR, 3: FATAL
import tensorflow as tf
tf.get_logger().setLevel('ERROR')

# 재현성을 위한 시드 고정
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

# Dataset 구성
dataset = tf.data.Dataset.from_tensor_slices((x_data, y_data)).batch(len(x_data))
print("dataset : ", dataset)

# 전처리 함수
def preprocess_data(features, labels):
    features = tf.cast(features, tf.float32)
    labels = tf.cast(labels, tf.float32)
    print("features : ", features)
    print("labels : ", labels)
    return features, labels


# 가중치 / 편향
W1 = tf.Variable(tf.random.normal([2, 1]), name="weight1")
b1 = tf.Variable(tf.random.normal([1]), name="bias1")

W2 = tf.Variable(tf.random.normal([2, 1]), name="weight2")
b2 = tf.Variable(tf.random.normal([1]), name="bias2")

W3 = tf.Variable(tf.random.normal([2, 1]), name="weight3")
b3 = tf.Variable(tf.random.normal([1]), name="bias3")


def neural_net(features):
    # 첫 번째 로지스틱 유닛
    layer1 = tf.sigmoid(tf.matmul(features, W1) + b1)
    # 두 번째 로지스틱 유닛
    layer2 = tf.sigmoid(tf.matmul(features, W2) + b2)
    # 두 유닛의 출력을 concat
    layer = tf.concat([layer1, layer2], axis=-1)  # shape: [batch, 2]
    # 최종 출력
    hypothesis = tf.sigmoid(tf.matmul(layer, W3) + b3)
    return hypothesis


def loss_fn(hypothesis, labels):
    # Binary cross entropy
    cost = -tf.reduce_mean(
        labels * tf.math.log(hypothesis + 1e-7)
        + (1.0 - labels) * tf.math.log(1.0 - hypothesis + 1e-7)
    )
    return cost


optimizer = tf.keras.optimizers.SGD(learning_rate=0.01)


def accuracy_fn(hypothesis, labels):
    predicted = tf.cast(hypothesis > 0.5, dtype=tf.float32)
    accuracy = tf.reduce_mean(tf.cast(tf.equal(predicted, labels), dtype=tf.float32))
    return accuracy


@tf.function
def train_step(features, labels):
    with tf.GradientTape() as tape:
        hypothesis = neural_net(features)
        loss_value = loss_fn(hypothesis, labels)
    grads = tape.gradient(loss_value, [W1, W2, W3, b1, b2, b3])
    optimizer.apply_gradients(zip(grads, [W1, W2, W3, b1, b2, b3]))
    return loss_value


EPOCHS = 50000

for step in range(EPOCHS):
    for features, labels in dataset:
        features, labels = preprocess_data(features, labels)
        loss_value = train_step(features, labels)

    if step % 5000 == 0:
        print(f"Iter: {step}, Loss: {loss_value.numpy():.4f}")

# 최종 정확도 확인
x_test, y_test = preprocess_data(x_data, y_data)
test_hypothesis = neural_net(x_test)
test_acc = accuracy_fn(test_hypothesis, y_test)
print(f"Testset Accuracy: {test_acc.numpy():.4f}")