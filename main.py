from voice_recorder import voiceRecorder
import text_preprocessor
import dataset_pre_processor
import pandas as pd

datasetPath = "D:\\dataset.xlsx"
dictionnaryPath = "D:\\Lexique.xlsx"
dictionnay = pd.read_excel(dictionnaryPath)
dataset = pd.read_excel(datasetPath)

# faire le preprocess du dataset pour pouvoir les utiliser pour les predictions
print("preprocessing datasets ...")
datasetPreprocessor = dataset_pre_processor.DatasetPreprocessor(dictionnay, dataset, 0.8)

"""MyRecorder = voiceRecorder()
output = MyRecorder.record_audio()
print(text_preprocessor.preprocess(output))"""
