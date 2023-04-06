import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from tensorflow import keras
import os


filedir = os.getcwd()
img = Image.open("demo_img2.jpeg").resize((64, 64)).convert("RGB")
Model = keras.models.load_model(f"{filedir}/network/model0.h5")

# print(len(keras.utils.img_to_array(img)))
pred = Model.predict(np.array([keras.utils.img_to_array(img) / 255.0]))
plt.imshow(pred[0])
plt.imshow(img)
plt.show()

