import os
import keras.models
import wandb
from keras import layers
import tensorflow as tf
from wandb.keras import WandbCallback


class Model:
    def __init__(self, inputLength, nb_categ, datasetpreprocessor):
        self.datasetpreprocessor= datasetpreprocessor
        self.n = nb_categ
        self.inputLength = inputLength
        self.model = keras.models.Sequential()
        self.model = self.create_model()

    def create_model(self):
        units = [128, 64, 32]
        rnn_input = self.model
        rnn_input.add(layers.Embedding(self.datasetpreprocessor.num_unique_words, 32, input_length=self.inputLength))
        for i in range(len(units)):
            rnn_input.add(layers.Bidirectional(keras.layers.LSTM(units[i], return_sequence=True, dropout=0.25),
                                               name="Bidirectional_layer" + str(i))())
        for i in range(len(units)):
            rnn_input.add(layers.Dense(units[i],activation="relu"))

        rnn_input.add(layers.Dense(self.n,activation="softmax"))

        opt= keras.optimizers.Adam(lr=0.001, beta_1=0.9 , beta_2=0.999,clipnorm=1.0)
        rnn_input.compile(optimizer=opt)
        rnn_input.summary()


        return rnn_input

    def train_model(self,epochs):
        early_stopping_patience=15
        early_stopping=keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=early_stopping_patience,
            restore_best_weights= True
        )

        checkpoint = tf.keras.callbacks.ModelCheckpoint(filepath=".\\Model\\ChatBotModel.hdf5",
                                                        monitor='val_loss',
                                                        verbose=1,
                                                        save_best_weights=True,
                                                        mode='min')
        wandb.login(key='54c3de5516f53236f9648bf7bdde028c4392b53b')
        wandb.init()
        callbacks_list=[checkpoint,
                        WandbCallback(monitor="val_loss",
                                      mode="min",
                                      log_weights=True),
                        early_stopping
                        ]
        if os.path.exists(".\\Model\\ChatBotModel.hdf5") :
            self.model=tf.keras.models.load_model(".\\Model\\ChatBotModel.hdf5")

        history= self.model.fit(
            self.datasetpreprocessor.train_padded,
            validation_data=self.datasetpreprocessor.val_padded,
            epochs=epochs,
            callbacks=callbacks_list
        )
        self.model.save(".\\Model\\ChatBotModel_final.h5")
        return history