import pickle, os
import matplotlib.pyplot as plt
from ann_visualizer.visualize import ann_viz
from tensorflow import keras

filedir = os.getcwd()

history = pickle.load(open(f"{filedir}/model/mod_history.pickle", "rb"))


for i, key in enumerate(history.keys()):
    plt.subplot(1, len(history.keys()), i) #select active subplot (it has 1 row, len(history) columns, and at the moment we are changing the ith plot)
    plt.plot(history[key])
    plt.xlabel("epoch")
    plt.ylabel(key)
    plt.legend("Model")
plt.title("Training data")
plt.show()