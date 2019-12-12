from __future__ import print_function, division

from keras.datasets import mnist
from keras.layers import Input, Dense, Reshape, Flatten, Dropout
from keras.layers import BatchNormalization, Activation, ZeroPadding2D
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import UpSampling2D, Conv2D
from keras.models import Sequential, Model
from keras.optimizers import Adam

from keras.wrappers.scikit_learn import KerasClassifier

import matplotlib.pyplot as plt
import sys
import numpy as np
np.set_printoptions(suppress=True, precision=4, threshold=100)
import os

# COMBO
sys.path.insert(1, "../COMBO")
from device_sequence_classifier import DeviceSequenceClassifier
import utils
import pandas as pd

# get COMBO data
def get_data(dataset_csv, use_cols, device, dsc):
    validation = utils.load_data_from_csv(dataset_csv, use_cols=use_cols)

    all_sess = dsc.split_data(validation)[0]
    other_dev_sess = validation.groupby(dsc.y_col).get_group(device)
    other_dev_sess = dsc.split_data(other_dev_sess)[0]

    classification = 1 if device == device else 0

    # get the optimal sequence length for data
    opt_seq_len = dsc.find_opt_seq_len(validation)
    seqs = []
    for i in range(opt_seq_len):
        seqs.append(other_dev_sess[i])

    # return sequences
    return seqs


# to pandas
def conv_to_pandas(array, columns):
    if array.shape != (1, 297):
        print("shape is", array.shape, "expected shape to be (1,297)")
        exit(1)
        
        
    dic = {}
    index = 0
    for column in columns:
        if column != "device_category":
            dic[column] = array[0][index]
            index += 1
        
    return pd.DataFrame(dic, index=[0])

# to numpy
def conv_to_numpy(array):
    return array.to_numpy()

# COMBO
def load_data(num):
    X_train = np.ones((num, 1, 297))
    print("\n\n x training shape",X_train.shape)
    
    return X_train

class GAN():

    # ---------------------
    # __init__
    # ---------------------
    def __init__(self):
        self.img_rows = 1
        self.img_cols = 297
        self.img_shape = (self.img_rows, self.img_cols)
        self.latent_dim = 297
        self.data_columns = pd.read_csv(os.path.abspath('../COMBO/data/use_cols.csv'))

        optimizer = Adam(0.0002, 0.5)

        # Build and compile the discriminator
        self.discriminator = self.build_discriminator("../COMBO/models/watch/watch_cart_entropy_100_samples_leaf.pkl")
        # Build the generator
        self.generator = self.build_generator()

        # The generator takes noise as input and generates samples

        # import pdb; pdb.set_trace()
        z = Input(shape=(self.latent_dim,))
        img = self.generator(z)
        
        # final GAN
        self.combined = Model(z, img)
        self.combined.compile(loss='binary_crossentropy', optimizer=optimizer)

    # ---------------------
    # add_start_data
    # ---------------------
    def add_start_data(self, data):
        self.start_data = data

    # ---------------------
    # add_start_data
    # ---------------------
    def load_data(self, batch_size):
        a = []
        for i in range(batch_size):
            a.append(conv_to_numpy(self.start_data[i]))

        return np.array(a)

    # ---------------------
    # build_generator
    # generates adversarial images
    # ---------------------
    def build_generator(self):

        model = Sequential()

        model.add(Dense(256, input_dim=self.latent_dim))
        model.add(LeakyReLU(alpha=0.2))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Dense(512))
        model.add(LeakyReLU(alpha=0.2))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Dense(1024))
        model.add(LeakyReLU(alpha=0.2))
        model.add(BatchNormalization(momentum=0.8))
        
        # activation needs to return positive outputs
        model.add(Dense(np.prod(self.img_shape), activation='softmax')) 
        model.add(Reshape(self.img_shape))

        model.summary()

        noise = Input(shape=(self.latent_dim,))
        img = model(noise)

        return Model(noise, img)


    # ---------------------
    # build_discriminator
    # used to build the discriminator and change it from an sklearn model to a 
    # keras model so that it can be trained with the discriminator
    # ---------------------
    def build_discriminator(self, discriminator_image):
        classifier = DeviceSequenceClassifier("../COMBO/models/", discriminator_image, use_cols="../COMBO/data/use_cols.csv", is_model_pkl=True)

        #import pdb; pdb.set_trace()

        return classifier.model

    # ---------------------
    # train
    # train the generator to learn from the output of the discriminator
    # ---------------------
    def train(self, epochs, batch_size=128, sample_interval=50):

        # Load the dataset
        X_train = self.load_data(batch_size)

        # Adversarial ground truths
        valid = np.ones((batch_size, 1, 297))
        fake = np.zeros((batch_size, 1, 297))

        for epoch in range(epochs):

            # ---------------------
            #  Get Sample
            # ---------------------

            # Select a random batch of samples
            idx = np.random.randint(0, X_train.shape[0], batch_size)
            samples = X_train[idx]

            noise = np.random.normal(0, 1, (batch_size, self.latent_dim))
            # noise = conv_to_numpy(self.start_data[0])

            # Generate a batch of new samples
            # Train on data that the discriminator was trained on
            generated_samples = self.generator.predict(noise)

            # ---------------------
            #  Determine accuracy
            # ---------------------
            accuracy_valid = []
            for sample in samples:
                pandafied = conv_to_pandas(sample, self.data_columns)
                predicted = self.discriminator.predict(pandafied)
                accuracy_valid.append(predicted)
                
            accuracy_fake = []
            for sample in generated_samples:
                pandafied = conv_to_pandas(sample, self.data_columns)
                predicted = self.discriminator.predict(pandafied)
                accuracy_fake.append(predicted)

            '''
            print("real samples", samples)
            print("real predictions", accuracy_valid)
            print("fake samples", generated_samples)
            print("fake predictions", accuracy_fake)
            '''
            
            d_loss = 0.5 * np.add(np.array(accuracy_valid), np.array(accuracy_fake))
            # 

            # ---------------------
            #  Train Generator
            # ---------------------

            noise = np.random.normal(0, 1, (batch_size, self.latent_dim))
            # noise = conv_to_numpy(self.start_data[0])

            # Train the generator
            g_loss = self.combined.train_on_batch(noise, valid)

            # If at save interval => save generated samples
            if epoch % sample_interval == 0:
                np.savetxt("samples/generated_%d.txt" % epoch, np.around(1000 * generated_samples[0], decimals=3), fmt='%d')
                print("%d [D loss: %f, acc.: %.2f%%] [G loss: %f]" % (epoch, d_loss[0], 100*d_loss[1], g_loss))  
                utils.eval_predictions(accuracy_valid, accuracy_fake)  



# ---------------------
# MAIN
# ---------------------
if __name__ == '__main__':
    # get data for watch
    use_cols = pd.read_csv(os.path.abspath('../COMBO/data/use_cols.csv'))
    watch_classifier = DeviceSequenceClassifier("../COMBO/models/", "../COMBO/models/watch/watch_cart_entropy_100_samples_leaf.pkl", use_cols="../COMBO/data/use_cols.csv", is_model_pkl=True)
    watch_data = get_data('../COMBO/data/validation.csv', pd.read_csv(os.path.abspath("../COMBO/data/use_cols.csv")), 'watch', watch_classifier)

    # test numpy / pandas stuff
    numpified = conv_to_numpy(watch_data[0]) # need shape 1, 297
    back = conv_to_pandas(numpified, use_cols) # will return 1, 297

    import pdb; pdb.set_trace()

    # train ?
    gan = GAN()
    gan.add_start_data(watch_data)
    gan.train(epochs=5, batch_size=5, sample_interval=5)
