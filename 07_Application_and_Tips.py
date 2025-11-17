import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"  # 0: INFO, 1: WARNING, 2: ERROR, 3: FATAL
import tensorflow as tf
import numpy as np
tf.get_logger().setLevel('ERROR')

# 4-1. sample data + normalization -------------------------------------------
# 예제 데이터 중간중간에 너무 큰 값들이 있다.
# => 이를 normalization을 통해 전처리 해 준다.
xy = np.array([
    [828.659973, 833.450012, 908100, 828.349976, 831.659973],
    [823.02002,  828.070007, 1828100, 821.655029, 828.070007],
    [819.929993, 824.400024, 1438100, 818.97998,  824.159973],
    [816.,       820.958984, 1008100, 815.48999,  819.23999],
    [819.359985, 823.,       1188100, 818.469971, 818.97998],
    [819.,       823.,       1198100, 816.,       820.450012],
    [811.700012, 815.25,     1098100, 809.780029, 813.669983],
    [809.51001,  816.659973, 1398100, 804.539978, 809.559998],
], dtype=np.float32)

# x(입력 특징들)과 y(타깃 값)를 분리
x_train_raw = xy[:, 0:-1]
y_train_raw = xy[:, [-1]]


# Normalization (0~1)
# x_new = (x - x_min) / (x_max - x_min)
def normalization(data):
    # 각 열(column)별로 min-max 정규화 (0~1 구간)
    numerator = data - np.min(data, 0)
    denominator = np.max(data, 0) - np.min(data, 0)
    return numerator / denominator


# xy 전체를 normalization 한 뒤 다시 x/y를 나눠도 됨
xy_norm = normalization(xy)
x_train = xy_norm[:, 0:-1]
y_train = xy_norm[:, [-1]]

# tf.data.Dataset 구성
dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train)).batch(len(x_train))


# 4-2. L2 Norm (모델, L2 loss 정의) ------------------------------------------
# 파라미터 초기화
W = tf.Variable(tf.random.normal([4, 1]), dtype=tf.float32, name="weight")
b = tf.Variable(tf.random.normal([1]), dtype=tf.float32, name="bias")


# 예측값 계산
def linearReg_fn(features):
    hypothesis = tf.matmul(features, W) + b
    return hypothesis


# 실제 loss에서 정규화된 값을 구하게 된다.
def l2_loss(loss, beta=0.01):
    # output = sum(t ** 2) / 2
    W_reg = tf.nn.l2_loss(W)
    loss = tf.reduce_mean(loss + W_reg * beta)
    return loss


# 실제 가설과 y값의 차이의 최소값을 구함.
# flag를 통해 L2 적용 여부를 결정함.
def loss_fn(hypothesis, labels, flag=False):
    cost = tf.reduce_mean(tf.square(hypothesis - labels))
    if flag:
        cost = l2_loss(cost)
    return cost


# 4-3. Learning Decay (학습 루프) -------------------------------------------
is_decay = True
starter_learning_rate = 0.1
EPOCHS = 101


if is_decay:
    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate=starter_learning_rate,
        decay_steps=50,   # decay_steps=50 과 동일 의미
        decay_rate=0.96,
        staircase=True,
    )
    optimizer = tf.keras.optimizers.SGD(learning_rate=lr_schedule)
else:
    optimizer = tf.keras.optimizers.SGD(learning_rate=starter_learning_rate)


def grad(features, labels, l2_flag):
    with tf.GradientTape() as tape:
        hypothesis = linearReg_fn(features)
        loss_value = loss_fn(hypothesis, labels, l2_flag)
    grads = tape.gradient(loss_value, [W, b])
    return grads, loss_value


def train():
    for step in range(EPOCHS):
        for features, labels in dataset:
            features = tf.cast(features, tf.float32)
            labels = tf.cast(labels, tf.float32)

            grads, loss_value = grad(features, labels, l2_flag=True)
            optimizer.apply_gradients(zip(grads, [W, b]))

        if step % 10 == 0:
            # 현재 learning rate 확인 (schedule 사용 시)
            if callable(optimizer.learning_rate):
                current_lr = optimizer.learning_rate(optimizer.iterations).numpy()
            else:
                current_lr = optimizer.learning_rate.numpy()

            print(
                "Iter: {}, Loss: {:.6f}, Learning Rate: {:.8f}".format(
                    step,
                    float(loss_value.numpy()),
                    float(current_lr),
                )
            )


if __name__ == "__main__":
    print("TensorFlow version:", tf.__version__)
    train()
    print("Trained W:", W.numpy())
    print("Trained b:", b.numpy())
