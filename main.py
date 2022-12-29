from voice_recorder import voiceRecorder
import text_preprocessor

import tensorflow as tf
import dataset_pre_processor
import pandas as pd

datasetPath = "D:\\dataset.xlsx"
dictionnaryPath = "D:\\Lexique.xlsx"
dictionnay = pd.read_excel(dictionnaryPath)
dataset = pd.read_excel(datasetPath)

# faire le preprocess du dataset pour pouvoir les utiliser pour les predictions
print("preprocessing datasets ...")
datasetPreprocessor = dataset_pre_processor.DatasetPreprocessor(dictionnay, dataset, 0.8)

# charger le model
chatModel = tf.keras.models.load_model(".\\Model\\ChatBotModel.hdf5")

# faire une prediction

while True:
    text = input("Question : ")
    text_ready = datasetPreprocessor.preprocess_text_to_predict(text)
    res = chatModel.predict(text_ready)
    print(res[0])


"""MyRecorder = voiceRecorder()
output = MyRecorder.record_audio()
print(text_preprocessor.preprocess(output))"""
