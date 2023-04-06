# from sklearn import metrics
from tensorflow import keras
from keras.layers import LeakyReLU, Conv2DTranspose, Conv2D, MaxPool2D, Reshape, Input, Dropout, UpSampling2D
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
        output = Reshape((128, 128, 3))(hidden_layers)
        

        #compile
        model = Model(inputs=[inputs], outputs=[output])
        model.compile(optimizer="Adam", loss="mse")

        return model
    

    #Convolution for CNN (convolution neural network)
    def Convolution(self, input_tensor, filters, kernel_size=(5, 5), pool_size=(2, 2), strides=(1, 1), name=""):
        
        x = Conv2D(filters, kernel_size, padding="same", strides=strides, use_bias=False)(input_tensor)
        x = Dropout(0.1)(x)
        if name:
             x = MaxPool2D(pool_size, name=name)(x)
        else:
             x = MaxPool2D(pool_size)(x)

        return x

    #hidden layers
    def hidden_layers(self, input):

        x = UpSampling2D((2, 2))(input)
        x = UpSampling2D((2, 2))(x)
        x = UpSampling2D((2, 2))(x)
        x = self.Convolution(x, 9)
        x = self.Convolution(x, 3)

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
