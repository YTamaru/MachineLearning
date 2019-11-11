import pandas as pd
import numpy as np
import cv2
from keras import backend as K
from keras.preprocessing.image import array_to_img, img_to_array, load_img
from keras.models import load_model

from tensorflow.python.keras.models import load_model

model = load_model('')#学習済みモデルの読み込み 
model.summary()

# モデルの最終出力を取り出す
model_output = model.output[:, 0]

# 最後の畳込み層を取り出す
last_conv = model.get_layer('')

from tensorflow.python.keras import backend as K 

grads = K.gradients(model_output, last_conv.output)[0]#最終畳み込み層の勾配

pooled_grads = K.mean(grads, axis=(0, 1, 2))

iterate = K.function([model.input],
                     [pooled_grads, last_conv.output[0]])

from tensorflow.python.keras.preprocessing import image
import numpy as np

img_path = './.jpg' #予測する画像
img_keras = image.load_img(img_path, target_size=(224, 224))
img_tensor = image.img_to_array(img_keras)
img_tensor = np.expand_dims(img_tensor, axis=0)
predicts = model.predict(img_tensor, batch_size=16, verbose=1, steps=None)
scenes_num = np.argmax(predicts)
scenes_class = [] #class名を入れる
print("The prediction is {}".format(scenes_class[scenes_num]) )#予測結果の出力

# モデルの訓練時と同じ方法で前処理
img_tensor /= 255.

pooled_grads_val, conv_output_val = iterate([img_tensor])#入力画像を関数に入れて、入力画像に対する最終畳み込み層出力の値と勾配を求める

for i in range(pooled_grads_val.shape[0]):
    conv_output_val[:, :, i] *= pooled_grads_val[i]

heatmap = np.mean(conv_output_val, axis=-1)

import cv2

# ヒートマップの後処理
heatmap = np.maximum(heatmap, 0)
heatmap /= np.max(heatmap)

# cv2を使って元画像を読み込む
img = cv2.imread(img_path)

# 元の画像と同じサイズになるようにヒートマップのサイズを変更
heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))

# ヒートマップをRGBに変換
heatmap = np.uint8(255 * heatmap)

# ヒートマップを元画像に適用
heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

# 0.4はヒートマップの強度係数
superimposed_img = heatmap * 0.4 + img

# 画像を保存
cv2.imwrite('heatmap.jpg', superimposed_img)
