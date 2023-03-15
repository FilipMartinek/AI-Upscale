# from sklearn import metrics
from tensorflow import keras
from keras.layers import BatchNormalization, LeakyReLU, Conv2DTranspose, Dropout, Reshape, Input
from keras.models import Model
# from tensorflow.keras.regularizers import l2
from keras.metrics import Precision

class MyModel:

    def assemble_model(self):
        #input layer
        input_shape = (64, 64, 3)
        inputs = Input((input_shape))

        #hidden layers
        hidden_layers = self.hidden_layers(inputs)

        #output layer
        output = Conv2DTranspose(3, (5, 5), padding="same", use_bias=False, activation="sigmoid", name="output")(hidden_layers)
        

        #compile
        model = Model(inputs=[inputs], outputs=[output])
        model.compile(optimizer="Adam", loss="mse", metrics=Precision())

        return model
    

    #Convolution for CNN (convolution neural network)
    def Convolution(self, input_tensor, filters, kernel_size=(5, 5), pool_size=(2, 2), strides=(1, 1), name=""):
        
        x = Conv2DTranspose(filters, kernel_size, padding="same", strides=strides, use_bias=False)(input_tensor)
        x = Dropout(0.1)(x)
        x = BatchNormalization()(x)
        if name:
            x = LeakyReLU(name=name)(x)
        else:
            x = LeakyReLU()(x)

        return x

    #hidden layers
    def hidden_layers(self, input):
        
        x = self.Convolution(input, 6)
        x = self.Convolution(x, 12)
        x = self.Convolution(x, 24)
        x = self.Convolution(x, 32)
        x = self.Convolution(x, 64)
        x = Reshape((128, 128, 16))(x)

        return x


#method to get a list of different models
def get_models():

    #return a list of models
    model = MyModel()
    return [
        model.assemble_model()
    ]


#if program is ran by itself, print the model summaries
if __name__ == "__main__":

    for model in get_models():
        model.summary()
