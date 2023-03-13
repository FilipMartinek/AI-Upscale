# from sklearn import metrics
from tensorflow import keras
from keras.layers import BatchNormalization, LeakyReLU, Conv2DTranspose, Dropout, Reshape, Input
from keras.models import Model
# from tensorflow.keras.regularizers import l2

class MyModel:

    def assemble_model(self):
        #input layer
        input_shape = (64, 64, 3)
        inputs = Input((input_shape))

        #hidden layers
        hidden_layers = self.hidden_layers(inputs)

        #output layer
        x = Conv2DTranspose(1, (5, 5), padding="same", use_bias=False, activation="tanh")(inputs)
        output = Reshape((64, 64, 3))

        #compile
        model = Model(inputs=[inputs], outputs=[output])
        model.compile(optimizers="Adam", loss=["binary_crossentropy"], metrics={"output":"accuracy"})

        return model
    

    #Convolution for CNN (convolution neural network)
    def Convolution(input_tensor, filters, kernel_size=(5, 5), pool_size=(2, 2), strides=(1, 1), name=""):
        
        print(f"---------------input_tensor: {input_tensor}, filters: {filters}")
        x = Conv2DTranspose(filters, kernel_size, padding="same", strides=strides, use_bias=False)(input_tensor)
        x = Dropout(0.1)(x) 
        x = BatchNormalization()(x)
        if name:
            x = LeakyReLU(name=name)(x)
        else:
            x = LeakyReLU()(x)

        return x

    def hidden_layers(self, input):
        #hidden layers
        print(f"---------------input_tensor: {input}, filters: {64}")
        x = self.Convolution(input_tensor=input, filters=64)
        x = self.Convolution(input_tensor=x, filters=86)
        x = self.Convolution(input_tensor=x, filters=108)

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
        model = model[0]
        model.summary()
