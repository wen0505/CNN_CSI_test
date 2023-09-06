import tensorflow as tf
import pandas as pd
import matplotlib.pyplot as plt

# 讀取 csv 檔
df = pd.read_csv('CSI_AMPLITUDE.csv')

# 將 manned 與 unmanned 的 CSI_AMPLITUDE 合併
# result = pd.concat([df[['CSI_AMPLITUDE']].rename(columns={'CSI_AMPLITUDE': 'manned'}),
#                     df[['CSI_AMPLITUDE']].rename(columns={'CSI_AMPLITUDE': 'unmanned'})], axis=1)

# 提取有人和沒人的 CSI_AMPLITUDE 列
people_csi = pd.DataFrame(data=df, columns=['manned'])
no_people_csi = pd.DataFrame(data=df, columns=['unmanned'])

# 將 dataframe 轉換成 dataset
people_ds = tf.data.Dataset.from_tensor_slices(people_csi.values)
no_people_ds = tf.data.Dataset.from_tensor_slices(no_people_csi.values)
# print(people_ds)

# 將 dataset 分為 80% 訓練、5% 驗證、15% 測試
people_train_ds, people_val_ds, \
    people_test_ds = tf.data.Dataset.split(people_ds.map(lambda x: x), (0.8, 0.05, 0.15))
no_people_train_ds, no_people_val_ds, \
    no_people_test_ds = tf.data.Dataset.split(no_people_ds.map(lambda x: x), (0.8, 0.05, 0.15))


# 轉換資料為 one-hot vector
people_ds = people_ds.map(lambda x: tf.one_hot(x, depth=2))
no_people_ds = no_people_ds.map(lambda x: tf.one_hot(x, depth=2))

# 將訓練、驗證、測試資料合併
train_ds = people_train_ds.concatenate(no_people_train_ds)
val_ds = people_val_ds.concatenate(no_people_val_ds)
test_ds = people_test_ds.concatenate(no_people_test_ds)

# 建立模型
model = tf.keras.Sequential()

# 添加卷積層
model.add(tf.keras.layers.Conv1D(32, 3, activation='relu', input_shape=(28, 1)))
model.add(tf.keras.layers.MaxPooling1D(2))
model.add(tf.keras.layers.Conv1D(64, 3, activation='relu'))
model.add(tf.keras.layers.MaxPooling1D(2))
model.add(tf.keras.layers.Conv1D(64, 3, activation='relu'))

# 添加平坦層
model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dense(64, activation='relu'))

# 添加輸出層
model.add(tf.keras.layers.Dense(2, activation='softmax'))

# 編譯模型
model.compile(optimizer='adam',
              loss=tf.keras.losses.CategoricalCrossentropy(),
              metrics=['accuracy'])
# 訓練模型
history = model.fit(train_ds, epochs=10, validation_data=val_ds)

# 顯示測試後的精確度數值
test_loss, test_acc = model.evaluate(test_ds, return_dict=True)
print('Test Accuracy:', test_acc)

# 繪製分布圖

plt.plot(history.history['accuracy'], label='accuracy')
plt.plot(history.history['val_accuracy'], label='val_accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.ylim([0.5, 1])
plt.legend(loc='lower right')

test_loss, test_acc = model.evaluate(test_ds)
print(test_acc)
