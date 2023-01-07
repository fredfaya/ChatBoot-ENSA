from Model import Model
import dataset_pre_processor
import pandas as pd
import online_data_service

# lecture des datasets
print("reading datasets ...")
dictionnay = dataset = online_data_service.get_data_from_sheet(online_data_service.LexiqueGShetName, online_data_service.TabName)
dataset = online_data_service.get_data_from_sheet(online_data_service.GSheetName, online_data_service.TabName)

# faire le preprocess du dataset
print("preprocessing datasets ...")
datasetPreprocessor = dataset_pre_processor.DatasetPreprocessor(dictionnay, dataset, 0.9)

# creation du model
print("create the model ...")
chatModel = Model(25, dataset["target"].value_counts().count(), datasetPreprocessor)

# entrainer le model
print("\n\nstart training the model ...\n\n")
chatModel.train_model(50000)

print("model training finish .")
