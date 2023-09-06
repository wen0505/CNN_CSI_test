import tensorflow as tf
import pandas as pd
import matplotlib.pyplot as plt
# https://blog.csdn.net/qq_39096123/article/details/100575784
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# 讀取 csv 檔
df = pd.read_csv('../src_SVM/p-np.csv')

# 將 manned 與 unmanned 的 CSI_AMPLITUDE 合併
# result = pd.concat([df[['CSI_AMPLITUDE']].rename(columns={'CSI_AMPLITUDE': 'manned'}),
#                     df[['CSI_AMPLITUDE']].rename(columns={'CSI_AMPLITUDE': 'unmanned'})], axis=1)

# 提取有人和沒人的 CSI_AMPLITUDE 列
no_people_csi = pd.DataFrame(data=df, columns=['unmanned'])
people_csi = pd.DataFrame(data=df, columns=['manned'])

# 將 dataframe 轉換成 dataset
no_people_ds = tf.data.Dataset.from_tensor_slices(no_people_csi.values)
people_ds = tf.data.Dataset.from_tensor_slices(people_csi.values)

# 將 dataset 分為 80% 訓練、5% 驗證、15% 測試
no_people_train_ds, no_people_val_ds, \
    no_people_test_ds = tf.data.Dataset.split(no_people_ds.map(lambda x: x), (0.8, 0.05, 0.15))
people_train_ds, people_val_ds, \
    people_test_ds = tf.data.Dataset.split(people_ds.map(lambda x: x), (0.8, 0.05, 0.15))
