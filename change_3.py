# import enssential models

from __future__ import print_function, division

from keras.datasets import mnist
from keras.layers import Input, Dense, Reshape, Flatten, Dropout, multiply
from keras.layers import BatchNormalization, Activation, Embedding, ZeroPadding2D
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import UpSampling2D, Conv2D
from keras.models import Sequential, Model , load_model
from keras.optimizers import Adam

import os
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np

from keras.preprocessing.image import array_to_img, img_to_array, load_img, save_img
import os, glob, shutil
from tqdm import tqdm

class CGAN_changed3():
    def __init__(self):
        # Input shape
        self.img_rows = 150 # the height of the input image
        self.img_cols = 150   # the width of the input image
        self.channels = 1   # channel = 1
        self.img_shape = (self.img_rows, self.img_cols, self.channels)  # the shape of single image is (64,64,1)
        self.num_classes = 2   # set the classes to 2, because the network will just classify fake and real images
        self.latent_dim = 256
        self.gdrive_path = '/content/gdrive'

        optimizer = Adam(0.0002, 0.5)   
        optimizer_g = Adam(0.002, 0.5) # we will set the second optimizer for generator for jusitification
        # Build and compile the discriminator
        self.discriminator = self.build_discriminator()     
        self.discriminator.compile(loss=['binary_crossentropy'],    
            optimizer=optimizer,
            metrics=['accuracy'])

        # Build the generator
        self.generator = self.build_generator()

        # The generator takes noise and the target label as input
        # and generates the corresponding digit of that label
        noise = Input(shape=(self.latent_dim,))     
        label = Input(shape=(1,))   
        img = self.generator([noise, label])    

        # For the combined model we will only train the generator
        self.discriminator.trainable = False  

        # The discriminator takes generated image as input and determines validity
        # and the label of that image
        valid = self.discriminator([img, label])   

        # The combined model  (stacked generator and discriminator)
        # Trains generator to fool discriminator
        self.combined = Model([noise, label], valid) 
        self.combined.compile(loss=['binary_crossentropy'],
            optimizer=optimizer_g)

    def build_generator(self):

        model = Sequential()    

        model.add(Dense(256, input_dim=self.latent_dim))   
        model.add(LeakyReLU(alpha=0.2))     
        model.add(BatchNormalization(momentum=0.8))     
        model.add(Dense(512))   
        model.add(LeakyReLU(alpha=0.2))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Dense(512))   
        model.add(LeakyReLU(alpha=0.2))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Dense(1024)) 
        model.add(LeakyReLU(alpha=0.2))
        model.add(BatchNormalization(momentum=0.8))

        model.add(Dense(np.prod(self.img_shape), activation='tanh')) 
        model.add(Reshape(self.img_shape))

        model.summary()

        noise = Input(shape=(self.latent_dim,))     
        label = Input(shape=(1,), dtype='int32')    
        label_embedding = Flatten()(Embedding(self.num_classes, self.latent_dim)(label))  

        model_input = multiply([noise, label_embedding])    
        img = model(model_input)    

        return Model([noise, label], img)

    def build_discriminator(self):

        model = Sequential()

        model.add(Dense(512, input_dim=np.prod(self.img_shape))) 
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dense(512))   
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.4))     
        model.add(Dense(512))   
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.4))
        model.add(Dense(512))   
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.4))
        model.add(Dense(1, activation='sigmoid'))  
        model.summary()

        img = Input(shape=self.img_shape)       
        label = Input(shape=(1,), dtype='int32')    

        label_embedding = Flatten()(Embedding(self.num_classes, np.prod(self.img_shape))(label))    
        flat_img = Flatten()(img)       

        model_input = multiply([flat_img, label_embedding])     

        validity = model(model_input)

        return Model([img, label], validity)

    def train(self, epochs, batch_size=128, sample_interval=50):

        (X_train, y_train) = self.preprocess_data()

        # create a directory to save the images
        img_dir = r"/content/images_change3"
        if not os.path.exists(img_dir):
            os.makedirs(img_dir)
        else:
            shutil.rmtree(img_dir)
            os.makedirs(img_dir)

        # Adversarial ground truths
        valid = np.ones((batch_size, 1))        
        fake = np.zeros((batch_size, 1))      

        for epoch in range(epochs):     

            # ---------------------
            #  Train Discriminator
            # ---------------------

            # Select a random half batch of images
            idx = np.random.randint(0, X_train.shape[0], batch_size)        
            imgs, labels = X_train[idx], y_train[idx]   # imgs.shape --> (batch_size,150,150,1)

            # Sample noise as generator input
            noise = np.random.normal(0, 1, (batch_size, self.latent_dim))   

            # Generate a half batch of new images
            gen_imgs = self.generator.predict([noise, labels])  

            # Train the discriminator
            d_loss_real = self.discriminator.train_on_batch([imgs, labels], valid)  
            d_loss_fake = self.discriminator.train_on_batch([gen_imgs, labels], fake)   
            d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)     

            # ---------------------
            #  Train Generator
            # ---------------------

            # Condition on labels
            sampled_labels = np.random.randint(0, 2, batch_size).reshape(-1, 1)   

            # Train the generator
            g_loss = self.combined.train_on_batch([noise, sampled_labels], valid)   

            # Plot the progress
            print ("%d [D loss: %f, acc.: %.2f%%] [G loss: %f]" % (epoch, d_loss[0], 100*d_loss[1], g_loss))

            # If at save interval => save generated image samples
            if epoch % sample_interval == 0 or epoch == epochs - 1:
                self.sample_images(epoch, epochs, img_dir)

    def sample_images(self, epoch, epochs, img_dir):
        if epoch == epochs-1:
            # save 50 images
            final_img_dir = os.path.join(self.gdrive_path, 'MyDrive', 'cgan_Assessment', 'Generated_covid_imgs_change3')
            if not os.path.exists(final_img_dir):
                os.mkdir(final_img_dir)

            noise = np.random.normal(0, 1, (50, self.latent_dim))
            gen_labels = np.zeros(50).reshape(-1, 1)
            gen_images = self.generator.predict([noise, gen_labels])

            img_nums = len(gen_images)

            for i in range(img_nums):
                # imgFileName = os.path.join(final_img_dir, 'epoch'+str(epoch) + '_' + str(i) + '.png')
                imgFileName = os.path.join(final_img_dir, '_' + str(i) + '.png')
                save_image = (gen_images[i])
                save_img(imgFileName, save_image)
        else:
            r, c = 2, 5
            noise = np.random.normal(0, 1, (10, self.latent_dim))
            gen_labels = np.zeros(10).reshape(-1, 1)
            gen_imgs = self.generator.predict([noise, gen_labels])

            gen_imgs = 0.5 * gen_imgs + 0.5
            fig, axs = plt.subplots(r, c)
            cnt = 0
            for i in range(r):
                for j in range(c):
                    axs[i][j].imshow(gen_imgs[cnt,:,:,0], cmap='gray')
                    axs[i][j].set_title("Image: %d" % cnt)
                    axs[i][j].axis('off')
                    cnt+= 1
            fig.savefig("images_change3/%d.png" % epoch)
            plt.close()
    def preprocess_data(self):
        img_dic = os.path.join(self.gdrive_path, 'MyDrive', 'cgan_Assessment', 'Covid-19')

        imgList = []
        imgList.extend(glob.glob(os.path.join(img_dic, '*')))

        y_train = np.array([0] * len(imgList))
        x_train  = np.zeros(shape=[len(imgList), 150, 150,1])

        for i in range(len(imgList)):
            img = load_img(imgList[i], color_mode='grayscale', target_size=(150, 150))
            img_array = img_to_array(img)
            img_array = img_array.astype(np.float64)

            x_train[i] = img_array

        # configure input
        x_train = (x_train.astype(np.float32) - 127.5) / 127.5
        y_train = y_train.reshape(-1, 1)

        return x_train, y_train
