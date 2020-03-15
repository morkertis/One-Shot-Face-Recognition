from sklearn.datasets import fetch_lfw_pairs
from keras.models import Model, Sequential
from keras.layers import Conv2D, MaxPool2D, Flatten, Dense, Input, Subtract, Lambda,Dropout,Concatenate
from keras.optimizers import Adam, SGD , RMSprop
from keras.regularizers import l2
import keras.backend as K
import numpy as np
from keras.callbacks import EarlyStopping , ModelCheckpoint



class SiameseNetwork:
    """
    Class that represents an architecture of deep learning model that implements Siamese Network
    The network is implemented based on the following article:
    Siamese Neural Networks for One-shot Image Recognition
    """
    def __init__(self, batch_size=64, resize=0.3, validation_per=0.1, epochs=10, iterations=5):
        """
        Class that handles creating new model, retreiving dataset (train and test),
        training and testing the model.
        :param batch_size: choose the size of batch for the training process
        :param resize: resize the images (float between 0 and 1)
        :param validation_per: percent of the validation set
        :param epochs: number of epoch for each iteration
        :param iterations: number of iterations
        """
        self.lfw_pairs_train = fetch_lfw_pairs(subset='train', funneled=False, resize=resize, slice_=None, color=False)
        self.lfw_pairs_train.pairs = self.lfw_pairs_train.pairs / 255

        shape = self.lfw_pairs_train.pairs.shape[2]
        K.clear_session()

        learning_rate = 10e-4
        # l2-regularization penalization for each layer
        self.l2_penalization = self.def_penalization()
        self.input_shape = (shape, shape, 1)
        convolutional_net = self.create_conv_net()

        # Now the pairs of images
        input_image_1 = Input(self.input_shape)
        input_image_2 = Input(self.input_shape)
        encoded_image_1 = convolutional_net(input_image_1)
        encoded_image_2 = convolutional_net(input_image_2)

        # L1 distance layer between the two encoded outputs
        # One could use Subtract from Keras, but we want the absolute value
        # l1_distance_layer = Lambda(lambda tensors: K.abs(tensors[0] - tensors[1]))
        distance = Lambda(euclidean_distance, output_shape=eucl_dist_output_shape)([encoded_image_1, encoded_image_2])
        self.model = Model([input_image_1, input_image_2], distance)
        # l1_distance = l1_distance_layer([encoded_image_1, encoded_image_2])
        rms = RMSprop()
        opt = Adam(lr=learning_rate)
        self.model.compile(loss=contrastive_loss, optimizer=rms, metrics=['binary_accuracy'])

        self.train_network(batch_size, epochs, iterations,validation_per)

        lfw_pairs_test = fetch_lfw_pairs(subset='test', funneled=False, resize=shape / 250, slice_=None, color=False)
        lfw_pairs_test.pairs = lfw_pairs_test.pairs / 255
        x_test, y_test, _, _ = train_validation_pairs(lfw_pairs_test, valid=0)
        self.model.evaluate(x_test, y_test)
        # save_model(model,'0.61_all_positive_10iter_10epoch_128_batch')


    def create_conv_net(self):
        """
        function that builds the convolutional network
        Number of layers/layers activations/filter sizes may vary
        -Hardcoded- #TODO - Think of a design that can change for experiments
        :return:
        """
        convolutional_net = Sequential()

        convolutional_net.add(Conv2D(filters=32, kernel_size=(10, 10),
                                     activation='relu',
                                     input_shape=self.input_shape,
                                     kernel_regularizer=l2(
                                         self.l2_penalization['Conv1']),
                                     name='Conv1'))

        convolutional_net.add(MaxPool2D())

        convolutional_net.add(Conv2D(filters=64, kernel_size=(7, 7),
                                     activation='relu',
                                     kernel_regularizer=l2(
                                         self.l2_penalization['Conv2']),
                                     name='Conv2'))
        convolutional_net.add(MaxPool2D())

        convolutional_net.add(Flatten())

        convolutional_net.add(
            Dense(units=100, activation='sigmoid', kernel_regularizer=l2(self.l2_penalization['Dense1']), name='Dense1'))
        return convolutional_net

    def def_penalization(self):
        """
        Define the penalization error (regularization) for each layer
        :return:
        """
        l2_penalization = {}
        l2_penalization['Conv1'] = 1e-2
        l2_penalization['Conv2'] = 1e-2
        l2_penalization['Conv3'] = 1e-2
        l2_penalization['Conv4'] = 1e-2
        l2_penalization['Dense1'] = 1e-4
        return l2_penalization

    def train_network(self, batch_size, epochs, iterations, validation_per):
        """
        Train SSN according to received parameters
        :param batch_size: choose the size of batch for the training process
        :param validation_per: percent of the validation set
        :param epochs: number of epoch for each iteration
        :param iterations: number of iterations
        :return:
        """
        best_model = 'models/best_model.h5'
        early_stopping_monitor = EarlyStopping(patience=5)
        checkpoint = ModelCheckpoint(filepath=best_model, monitor='val_loss', save_best_only=True)
        batch_size = batch_size
        epochs = epochs
        iteration = iterations
        percents = np.linspace(0.8, 0.2, iteration)

        # %%
        hist = []
        x_train, y_train, x_valid, y_valid = train_validation_pairs(self.lfw_pairs_train, valid=validation_per)
        for i in range(iteration):
            print(i)
            percent = percents[i]
            if (i + 1) % 3 == 0:
                K.set_value(self.model.optimizer.lr, K.get_value(
                    self.model.optimizer.lr) * 0.95)
            x_train_percent, y_train_percent = persons_same_different_split(x_train, y_train, percent_same=percent)
            history = self.model.fit(x=x_train_percent, y=y_train_percent, validation_data=(x_valid, y_valid), epochs=epochs,
                                batch_size=batch_size, verbose=True, callbacks=[early_stopping_monitor, checkpoint])
            hist.append(history)
            self.model.load_weights(best_model)


def train_validation_pairs(lfw_pairs_train, valid=0.2):
    """
    Split to train and validation set
    Function is set to randomly choose equal number of same-person faces and differet-person faces
    :param lfw_pairs_train:
    :param valid:
    :return:
    """
    ln = len(lfw_pairs_train.target)

    indices_same = np.random.choice(np.argwhere(lfw_pairs_train.target == 1).flatten(), int((1-valid/2) * ln / 2))
    indices_different = np.random.choice(np.argwhere(lfw_pairs_train.target == 0).flatten(),
                                         int((1 - valid/2) * ln / 2))
    indices = np.concatenate((indices_same, indices_different), axis=0)
    x_train = lfw_pairs_train.pairs[indices]
    x_train = [np.expand_dims(x_train[:, 0], axis=3), np.expand_dims(x_train[:, 1], axis=3)]
    y_train = lfw_pairs_train.target[indices]

    not_train_indices = np.arange(ln)[~np.isin(np.arange(ln), indices)]
    indices_same_valid = np.random.choice(not_train_indices[np.where(not_train_indices < int(ln / 2))],
                                          int(valid / 2 * ln / 2))
    indices_different_valid = np.random.choice(not_train_indices[np.where(not_train_indices >= int(ln / 2))],
                                               int(valid / 2 * ln / 2))
    indices_valid = np.concatenate((indices_same_valid, indices_different_valid), axis=0)

    x_valid = lfw_pairs_train.pairs[indices_valid]
    x_valid = [np.expand_dims(x_valid[:, 0], axis=3), np.expand_dims(x_valid[:, 1], axis=3)]
    y_valid = lfw_pairs_train.target[indices_valid]

    return x_train, y_train, x_valid, y_valid


def persons_same_different_split(x_train, y_train, percent_same=0.5):
    """
    Function to split the training set on each iteration
    The function is set to un-equally choose all the same-face training samples,
    and percent of different-face samples (according to percent_same)
    :param x_train: X part of the training set
    :param y_train: classification of the training set
    :param percent_same: percent of different-face samples
    :return:
    """
    np.random.seed(int(percent_same * 100))
    ln = len(y_train)

    indices_same = np.random.choice(np.argwhere(y_train == 1).flatten(), int(1 * ln / 2))
    indices_different = np.random.choice(np.argwhere(y_train == 0).flatten(),
                                         int((1 - percent_same) * ln / 2))
    indices = np.concatenate((indices_same, indices_different), axis=0)
    x_train = [x_train[0][indices],x_train[1][indices]]
    y_train = y_train[indices]
    return x_train, y_train


def euclidean_distance(vects):
    x, y = vects
    return K.sqrt(K.maximum(K.sum(K.square(x - y), axis=1, keepdims=True), K.epsilon()))


def eucl_dist_output_shape(shapes):
    shape1, shape2 = shapes
    return (shape1[0], 1)


def contrastive_loss(y_true, y_pred):
    '''Contrastive loss from Hadsell-et-al.'06
    http://yann.lecun.com/exdb/publis/pdf/hadsell-chopra-lecun-06.pdf
    '''
    margin = 1
    return  K.mean((1 - y_true) * K.square(y_pred) + y_true * K.square(K.maximum(margin - y_pred, 0)))
