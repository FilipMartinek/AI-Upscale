import os, sys, pickle, threading
import numpy as np
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from datetime import datetime
import model, init_dataset

#USE GUIDE:
#python3.8 network/train.py [optional: MODEL_NUM] [optional: MAXLEN] [optional: GPU_NUM] [optional: "m" for multiple gpu use]
#GPU_NUM will be taken as number of gpus with "m" paramater


#select gpu
#os.environ["CUDA_VISIBLE_DEVICES"] = "0" #first gpu
#os.environ["CUDA_VISIBLE_DEVICES"] = "1" #second gpu


#create a filedir vat and get models
filedir = os.getcwd() + "/network/"
models = model.get_models()


#create vars for number of epochs and batch size
EPOCHS = 300
BATCH_SIZE = 32


#train, save model and history
def train(Model, input_train, input_test, output_train, output_test, filename):

    #create checkpointer and early stop, then add them to the callback list
    Checkpoint = ModelCheckpoint(filename, monitor="val_loss", verbose=1,save_best_only=True, save_weights_only=False, mode="auto", save_freq="epoch")
    Early_stop = EarlyStopping(patience=150, monitor="val_loss",restore_best_weights=True),
    callbacks = [Checkpoint, Early_stop]


    #train model and save training history
    start = datetime.now()
    History = Model.fit(input_train, output_train, batch_size=BATCH_SIZE, validation_data=(input_test, output_test), epochs=EPOCHS, callbacks=[callbacks])
    time = datetime.now() - start
    history = History.history
    history.update({"time" : time})
    pickle.dump(history, open(f"{filename[:3]}_history.pickle", "wb"), pickle.HIGHEST_PROTOCOL)

    #save complete model
    Model.save(filename)


def get_paramaters():
    #default parameters
    MAXLEN = 5000
    MULTIPLE_GPUS = False
    GPUNUM = "-1"
    MODEL_NUM = -1


    #try to get a model num paramater
    try:
        #get model num
        temp = 1
        MODEL_NUM = int(sys.argv[temp])
        temp += 1


        #try to get other paramaters
        
        try:
            MAXLEN = int(sys.argv[temp])
            temp += 1
        except IndexError or ValueError:
            pass
    
        try:
            GPUNUM = sys.argv[temp]
        except IndexError:
            pass

        #select gpu
        os.environ["CUDA_VISIBLE_DEVICES"] = GPUNUM

    except Exception:

        #try to get other paramaters
        try:
            MAXLEN = int(sys.argv[temp])
            temp += 1
        except IndexError or ValueError:
            pass
        try:
            GPUNUM = sys.argv[temp]
            temp += 1
        except IndexError:
            pass
        try:
            if sys.argv[temp] == "m":
                MULTIPLE_GPUS = True
        except IndexError:
            pass

    return MODEL_NUM, MAXLEN, MULTIPLE_GPUS, GPUNUM

#thread for training
def thread(models, gpu, data_len):

            #select gpu
            os.environ["CUDA_VISIBLE_DEVICES"] = gpu


            #go through all the models and train them
            for i, model_data in enumerate(models):
                
                #get model data
                print(model_data)
                Model = model_data
                input_train, input_test, output_train, output_test = init_dataset.get_data(data_len)

                #train, save, and evaluate model
                train(Model, input_train, input_test, output_train, output_test, f"model{i}")
                Model.save(f"determine_face/models/model{i}.h5")
                Model.evaluate(input_test,output_test)

#if program is ran
if __name__ == "__main__":
    
        MODEL_NUM, DATA_LEN, MULTIPLE_GPUS, GPUNUM = get_paramaters()

        #train everything by default
        if MODEL_NUM == -1: 
             #if there's only one gpu
            if not(MULTIPLE_GPUS):
                thread(models, GPUNUM, DATA_LEN)
            else:
                chunk = int(len(models)) // GPUNUM
                for chunknum, i in enumerate(range(0, len(models), chunk)):
                    #create and start a thread
                    x = threading.Thread(target=thread, args=[models[i:i+chunk], str(chunknum), DATA_LEN])
                    x.start()
                #last thread if models is not divisible by GPUNUM
                if int(len(models)) % GPUNUM > 0:
                    thread(models[i:])

        else:
            #get model data
            Model = models[MODEL_NUM]
            
            #train model
            thread([Model], "-1", DATA_LEN)        
        

        
