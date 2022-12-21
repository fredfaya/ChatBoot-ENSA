from voice_recorder import voiceRecorder
import text_preprocessor
import dataset_pre_processor
import pandas as pd
#import tensorflow as tf

"""MyRecorder = voiceRecorder()
output = MyRecorder.record_audio()
print(text_preprocessor.preprocess(output))"""
datasetPath = "C:\\Users\\Mehdi\\OneDrive\\Bureau\\Lexique.xlsx"
dataset = pd.read_excel(datasetPath)