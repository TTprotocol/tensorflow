import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"  # 0: INFO, 1: WARNING, 2: ERROR, 3: FATAL
import tensorflow as tf
tf.get_logger().setLevel('ERROR')
import numpy as np

tf.random.set_seed(777)

x_data = np.array([[0, 0],
                   [0, 1],
                   [1, 0],
                   [1, 1]], dtype=np.float32)

y_data = np.array([[0],
                   [1],
                   [1],
                   [0]], dtype=np.float32)

model = tf.keras.models.Sequential([
    tf.keras.layers.Dense(10, input_dim=2, activation=tf.nn.sigmoid),
    tf.keras.layers.Dense(10, activation=tf.nn.sigmoid),
    tf.keras.layers.Dense(10, activation=tf.nn.sigmoid),
    tf.keras.layers.Dense(1,  activation=tf.nn.sigmoid),
])

model.compile(optimizer="adam",
              loss="binary_crossentropy",
              metrics=["binary_accuracy"])

tb_hist = tf.keras.callbacks.TensorBoard(
    log_dir="./logs/xor_logs_keras",
    histogram_freq=0,
    write_graph=True,
    write_images=True,
)

model.fit(x_data, y_data, epochs=5000, callbacks=[tb_hist], verbose=0)

pred = model.predict(x_data)
pred_label = (pred > 0.5).astype(np.int32)

print("Predicted probability:\n", pred)
print("Predicted label:\n", pred_label)