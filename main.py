import tensorflow as tf
from Data import dataset_pre_processor
import pandas as pd
import numpy as np

print("reading datasets ...")
datasetPath = "D:\\dataset.xlsx"
dictionnaryPath = "D:\\Lexique.xlsx"
dictionnay = pd.read_excel(dictionnaryPath)
dataset = pd.read_excel(datasetPath)

# faire le preprocess du dataset pour pouvoir les utiliser pour les predictions
print("preprocessing datasets ...")
datasetPreprocessor = dataset_pre_processor.DatasetPreprocessor(dictionnay, dataset, 1)

# charger le model
print("Loading model ...")
chatModel = tf.keras.models.load_model(".\\Model\\ChatBotModel_V1.hdf5")

scores = chatModel.evaluate(datasetPreprocessor.train_padded, datasetPreprocessor.train_labels, verbose=0)
print('Accuracy on training data: {} \n Error on training data: {}'.format(scores[1], 1 - scores[1]))

# faire une prediction

while True:
    text = input("Question : ")
    text_ready = datasetPreprocessor.preprocess_text_to_predict(text)
    res = chatModel.predict(text_ready)
    print(np.argmax(res[0]))


"""MyRecorder = voiceRecorder()
output = MyRecorder.record_audio()
print(text_preprocessor.preprocess(output))"""
