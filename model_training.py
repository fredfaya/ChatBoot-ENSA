from Model import Model
import dataset_pre_processor
import pandas as pd

# lecture des datasets
print("reading datasets ...")
datasetPath = "D:\\dataset.xlsx"
dictionnaryPath = "D:\\Lexique.xlsx"
dictionnay = pd.read_excel(dictionnaryPath)
dataset = pd.read_excel(datasetPath)

# faire le preprocess du dataset
print("preprocessing datasets ...")
datasetPreprocessor = dataset_pre_processor.DatasetPreprocessor(dictionnay, dataset, 0.8)

# creation du model
print("create the model ...")
chatModel = Model(100, dataset["target"].value_counts().count(), datasetPreprocessor)

# entrainer le model
print("\n\nstart training the model ...\n\n")
chatModel.train_model(150)

print("model training finish .")
