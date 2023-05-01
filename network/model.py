# from sklearn import metrics
import tensorflow as tf
from tensorflow import keras
from keras.layers import LeakyReLU, Conv2DTranspose, Conv2D, MaxPool2D, Reshape, Input, Dropout, UpSampling2D, Dense, Flatten
from keras.models import Model
from keras.losses import BinaryCrossentropy
from keras.regularizers import l2
from tensorflow.keras.optimizers import Adam

class Models:

    #get generator model
    def assemble_generator(self):
        #input layer
        input_shape = (64, 64, 3)
        inputs = Input((input_shape))

        #hidden layers
        hidden_layers = self.gen_hidden_layers(inputs)

        #output layer
        output = Reshape((128, 128, 3))(hidden_layers)
        

        #create and return a tf Model object
        model = Model(inputs=[inputs], outputs=[output])

        return model

    #hidden layers
    def gen_hidden_layers(self, input):

        x = UpSampling2D((2, 2))(input)
        x = UpSampling2D((2, 2))(x)
        x = UpSampling2D((2, 2))(x)
        x = self.Convolution(x, 9)
        x = self.Convolution(x, 3)

        return x
    
    #get discriminator model
    def assemble_discriminator(self):
        #input layer
        input_shape = (128, 128, 3)
        inputs = Input((input_shape))

        #hidden layers
        hidden_layers = self.disc_hidden_layers(inputs)

        #output layer
        output = Dense(1, activation="sigmoid")(hidden_layers)
        

        #create and return a tf Model object
        model = Model(inputs=[inputs], outputs=[output])

        return model

    #hidden layers
    def disc_hidden_layers(self, input):
        x = self.Convolution(input, 6)
        x = self.Convolution(x, 12)
        x = self.Convolution(x, 12)
        x = self.Convolution(x, 6)
        x = Flatten()(x)
        x = Dense(128)(x)
        x = Dense(64)(x)
        x = Dense(8)(x)

        return x
    
    #Convolution for CNN (convolution neural network)
    def Convolution(self, input_tensor, filters, kernel_size=(5, 5), pool_size=(2, 2), strides=(1, 1), name=""):
        
        x = Conv2D(filters, kernel_size, padding="same", strides=strides, kernel_regularizer=l2(1e-4))(input_tensor)
        x = Dropout(0.3)(x)
        if name:
             x = MaxPool2D(pool_size, name=name)(x)
        else:
             x = MaxPool2D(pool_size)(x)

        return x

#class to conect the discriminator and generator objects and create a custom training loop
class GAN(Model):
    
    #initialize (you can select the models used in the siamese model here)
    def __init__(self, generator, discriminator, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.generator = generator
        self.discriminator = discriminator

    #compile (you can select the optimizers here)
    def compile(self, gen_opt=Adam(3e-4), disc_opt=Adam(3e-4), losses=BinaryCrossentropy()):
        super().compile()
        self.gen_opt = gen_opt
        self.disc_opt = disc_opt
        self.loss = losses


    #loss functions
    def gen_loss(self, disc_out_gen, data_in, gen_out):
        loss = self.loss(tf.zeros_like(disc_out_gen), disc_out_gen)
        
        data_in_slice = data_in[:, 0, 0, :]
        gen_out_slice = gen_out[:, 0, 0, :]
        # gen_out_slice = tf.keras.layers.Average()([gen_out[:, 0:128:2, 0:128:2, :], gen_out[:, 1:128:2, 0:128:2, :], gen_out[:, 0:128:2, 1:128:2, :], gen_out[:, 1:128:2, 1:128:2, :]]) #downscale upscaled image
        # print(tf.math.subtract(gen_out_slice, data_in_slice).numpy())
        loss2 = self.loss(data_in_slice, gen_out_slice) # for color accuracy (checks pixel colors for corner pixels)
        loss = tf.stack([loss, loss2], axis=0)
        return loss
    
    def disc_loss(self, real_out, gen_out):
        generated = tf.concat([real_out, gen_out], axis=0)
        correct = tf.concat([tf.zeros_like(real_out), tf.ones_like(gen_out)], axis=0)
        loss = self.loss(correct, generated)
        return loss
    
    #custom training loop. Trains both models at once.
    def train_step(self, data):
        real_out, data_in = data
        data_in = tf.reshape(data_in, (-1, 64, 64, 3))     #reshape to fit neural net even when batch size is 1 (dims would be missing first dimension)
        real_out = tf.reshape(real_out, (-1, 128, 128, 3)) #

        #get generator and discriminator outputs
        fake_out = self.generator(data_in, training=True)

        #get loss
        with tf.GradientTape() as disc_tape, tf.GradientTape() as gen_tape:

            #train discriminator and generator, and calculate losses

            #discriminator
            disc_out_real = self.discriminator(real_out, training=True)
            disc_out_fake = self.discriminator(fake_out, training=True)
            
            discriminator_loss = self.disc_loss(disc_out_real, disc_out_fake)

            #generator
            gen_pred = self.generator(data_in, training=True)
            disc_pred = self.discriminator(gen_pred, training=False)
            generator_loss = self.gen_loss(disc_pred, data_in, real_out)
        
        #calculate and apply gradient to network parameters
        disc_trainable_vars = self.discriminator.trainable_variables
        disc_grad = disc_tape.gradient(discriminator_loss, disc_trainable_vars)
        self.disc_opt.apply_gradients(zip(disc_grad, disc_trainable_vars))

        gen_trainable_vars = self.generator.trainable_variables
        gen_grad = gen_tape.gradient(generator_loss, gen_trainable_vars)
        self.gen_opt.apply_gradients(zip(gen_grad, gen_trainable_vars))

        return {"g_loss" : generator_loss, "d_loss" : discriminator_loss}

#method to get a list of different models
def get_model():

    #return a list of models
    models = Models()
    model = GAN(models.assemble_generator(), models.assemble_discriminator())
    model.compile()
    return model


#if program is ran by itself, print the model summaries
if __name__ == "__main__":

    Models().assemble_generator().summary()
    Models().assemble_discriminator().summary()
