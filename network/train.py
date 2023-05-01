import os, sys, pickle, threading
import numpy as np
import tensorflow as tf
from datetime import datetime
import model, init_dataset

#USE GUIDE:
#python3.8 network/train.py [optional: MAXLEN] [optional: GPU_NUM] [optional: "ow" for overwriting dataset]


#create a filedir var and get model
filedir = os.getcwd()
Model = model.get_model()


#create vars for number of epochs and batch size
EPOCHS = 100
BATCH_SIZE = 256

#train, save model and history
def train(Model, filename, gpu=0, data_len=10000, ow=False):

    #select gpu
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu

    #get dataset
    train_dataset = init_dataset.get_data(data_len, ow)



    #train model and save training history
    start = datetime.now()
    History = Model.fit(train_dataset, epochs=EPOCHS, batch_size=BATCH_SIZE)
    time = datetime.now() - start
    history = History.history
    history.update({"time" : time})
    pickle.dump(history, open(f"{filename[:3]}_history.pickle", "wb"), pickle.HIGHEST_PROTOCOL)

    #save complete model
    Model.generator.save(filename)
    Model.discriminator.save(filename[:3] + "_discriminator.h5")

#function to get paramater from sys.argv
def get_paramaters():
    #default parameters
    DATA_LEN = 10000
    GPUNUM = "0"
    OW = False

    #try to get the paramaters
    temp = 1
    try:
        DATA_LEN = int(sys.argv[temp])
        temp += 1
    except (IndexError, ValueError):
        pass

    try:
        GPUNUM = sys.argv[temp]
    except IndexError:
        pass

    if "ow" in sys.argv:
        OW = True

    

    return DATA_LEN, GPUNUM, OW

#if program is ran
if __name__ == "__main__":

        DATA_LEN, GPUNUM, OW = get_paramaters()

        train(Model, f"{filedir}/model/model.h5", GPUNUM, DATA_LEN, OW)    
        

        
