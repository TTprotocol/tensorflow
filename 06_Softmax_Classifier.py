import tensorflow as tf
import numpy as np
tf.get_logger().setLevel('ERROR')

# 1. 데이터 정의 (사용자 코드 유지)
x_data = [
    [1, 2, 1, 1],
    [2, 1, 3, 2],
    [3, 1, 3, 4],
    [4, 1, 5, 5],
    [1, 7, 5, 5],
    [1, 2, 5, 6],
    [1, 6, 6, 6],
    [1, 7, 7, 7],
]

# One-Hot encoding: 특정 부분에 대해서만 표기하고 나머지는 0으로 표기하는 방법.
y_data = [
    [0, 0, 1],
    [0, 0, 1],
    [0, 0, 1],
    [0, 1, 0],
    [0, 1, 0],
    [0, 1, 0],
    [1, 0, 0],
    [1, 0, 0],
]

# convert into numpy and float format
x_data = np.asarray(x_data, dtype=np.float32)
y_data = np.asarray(y_data, dtype=np.float32)

nb_classes = 3  # num classes

# 2. 가중치, 편향 정의
w = tf.Variable(tf.random.normal([4, nb_classes]), name='weight')
b = tf.Variable(tf.random.normal([nb_classes]), name='bias')     
variables = [w, b]

# 3. 가설 함수 정의
def hypothesis(X):
    # X: (batch_size, 4)
    logits = tf.matmul(X, w) + b
    return tf.nn.softmax(logits)

# 4. 비용 함수 정의
def cost_fn(X, Y):
    logits = hypothesis(X)  # (batch_size, nb_classes)
    # one-hot cross entropy
    # Y: one-hot 레이블, logits: softmax 확률
    cost = -tf.reduce_sum(Y * tf.math.log(logits + 1e-7), axis=1)
    cost_mean = tf.reduce_mean(cost)
    return cost_mean

# 5. 기울기 계산 함수
def grad_fn(X, Y):
    with tf.GradientTape() as tape:
        cost = cost_fn(X, Y)
    grads = tape.gradient(cost, variables)
    return grads

# 6. 학습 함수
def fit(X, Y, epochs=2000, verbose=100):
    optimizer = tf.keras.optimizers.SGD(learning_rate=0.1)

    for i in range(epochs):
        grads = grad_fn(X, Y)
        # grads: [dw, db], variables: [w, b]
        optimizer.apply_gradients(zip(grads, variables))

        if (i == 0) or ((i + 1) % verbose == 0):
            # .numpy()로 현재 cost 값을 출력
            print('Loss at epoch %d: %f' % (i + 1, cost_fn(X, Y).numpy()))

# 7. 학습 실행
fit(x_data, y_data, epochs=2000, verbose=100)

# 8. 학습된 모델로 예측 확인
a = hypothesis(x_data)

print(a)                      # 각 샘플에 대한 [class0, class1, class2] 확률
print(tf.argmax(a, 1))        # 예측된 class index
print(tf.argmax(y_data, 1))   # 실제 정답 class index (one-hot → index)

# 9. 새로운 입력(sample_db)에 대한 softmax 예측
sample_db = [[8, 2, 1, 4]]
sample_db = np.asarray(sample_db, dtype=np.float32)

print(hypothesis(sample_db))  # 예: tf.Tensor([[0.93, 0.06, 0.007]], ...)


# ----------------------------------------------------------------------------------------------------------------------------------

# # 수정된 코드
# import tensorflow as tf
# import numpy as np
# tf.get_logger().setLevel('ERROR')

# x_data = [[1, 2, 1, 1],
#     [2, 1, 3, 2],
#     [3, 1, 3, 4],
#     [4, 1, 5, 5],
#     [1, 7, 5, 5],
#     [1, 2, 5, 6],
#     [1, 6, 6, 6],
#     [1, 7, 7, 7]]
# y_data = [[0, 0, 1],
#     [0, 0, 1],
#     [0, 0, 1],
#     [0, 1, 0],
#     [0, 1, 0],
#     [0, 1, 0],
#     [1, 0, 0],
#     [1, 0, 0]]
# # 3개의 클래스 분류할때 0,1,2 를 각각 [1,0,0] , [0,1,0], [0,0,1]로 하나만 hot하게 표시함

# # np.float32 배열 전환
# x_data = np.asarray(x_data, dtype = np.float32)
# y_data = np.asarray(y_data, dtype = np.float32)

# # y의 개수 = 클래스 개수 = label 개수
# dataset = tf.data.Dataset.from_tensor_slices((x_data, y_data)).batch(len(x_data))
"""
dataset 설명
1. from_tensor_slices : 슬라이스 단위로 데이터를 쪼개서 dataset을 만든다.
    현재 튜플 (x_data, y_data) 를 넣었으므로, 구조는 다음과 같다.
    0번 : (x_data[0], y_data[0])
    1번 : (x_data[1], y_data[1])
    ...
    7번 : (x_data[7], y_data[7])
2. .batch(len(x_data))
    len(x_data)는 8이다.
    따라서 .batch(8)을 붙이면 8개 element를 한 번에 묶어서 하나의 배치로 만들어준다.
"""
# W = tf.Variable(tf.random.normal([4, 3], name='weight'))
# b = tf.Variable(tf.random.normal([3]), name = 'bias')
# variable = [W, b]

# dataset.element_spec

# # softmax 함수 정의
# def softmax_fn (features):
#     hypothesis = tf.nn.softmax(tf.matmul(features, W) + b)
#     return hypothesis

# # 손실함수 정의
# def loss_fn (features, labels):
#     hypothesis = tf.nn.softmax(tf.matmul(features, W) + b)
#     cost = tf.reduce_mean(-tf.reduce_sum(y_data * tf.math.log(hypothesis), axis = 1))
#     return cost

# # gradient 함수 정의
# def grad (hypothesis, features, labels):
#     with tf.GradientTape() as tape:
#         loss_value = loss_fn(features, labels)
#     return tape.gradient(loss_value, [W, b])

# # 확률적 경사 하강법(SGD) 학습률 0.01 사용
# optimizer = tf.keras.optimizers.SGD(learning_rate = 0.01)
    
# n_epochs = 3000
# for step in range(n_epochs + 1):
    
#     for features, labels in iter(dataset):
#         # hypothesis = softmax_regression (features)
#         hypothesis = softmax_fn(features)
#         grads = grad(hypothesis, features, labels)
#         optimizer.apply_gradients(grads_and_vars = zip(grads, [W, b]))
    
#     if step % 300 == 0:
#             # dataset에서 뽑은 전체 배치값의 평균 손실을 출력
#             print("iter: {}, Loss: {:.4f}".format(step, loss_fn(features, labels)))
            
# a = x_data
# a = softmax_fn(a)
# print(hypothesis) # softmax 함수를 통과시킨 x_data

# # argmax 가장 큰 값의 index를 찾아줌

# print(tf.argmax(a, 1)) # 가설을 통한 예측값
# print(tf.argmax(y_data, 1)) # 실제 값
