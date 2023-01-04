from flask import Flask, jsonify, request
from flask_cors import CORS
import tensorflow as tf
import dataset_pre_processor
import pandas as pd
import numpy as np

# liste des reponses appropriees a chaques questions
reponse_question_0 = "reponse a la question 0"
reponse_question_1 = "reponse a la question 1"
reponse_question_2 = "reponse a la question 2"
reponse_question_3 = "reponse a la question 3"
reponse_question_4 = "reponse a la question 4"
reponse_question_5 = "reponse a la question 5"
reponse_question_6 = "reponse a la question 6"
reponse_question_7 = "reponse a la question 7"
reponse_question_8 = "reponse a la question 8"
reponse_question_9 = "reponse a la question 9"
reponse_not_known = "Je n'ai pas la reponse a votre question; Pouvez vous reformuler s'il vous plait ?"

list_responses = [reponse_question_0,
                  reponse_question_1,
                  reponse_question_2,
                  reponse_question_3,
                  reponse_question_4,
                  reponse_question_5,
                  reponse_question_6,
                  reponse_question_7,
                  reponse_question_8,
                  reponse_question_9]

print("reading datasets ...")
datasetPath = "D:\\dataset.xlsx"
dictionaryPath = "D:\\Lexique.xlsx"
dictionary = pd.read_excel(dictionaryPath)
dataset = pd.read_excel(datasetPath)

# faire le preprocess du dataset pour pouvoir les utiliser pour les predictions
print("preprocessing datasets for text encoding...")
datasetPreprocessor = dataset_pre_processor.DatasetPreprocessor(dictionary, dataset, 1)

# charger le model
print("Loading model ...")
chatModel = tf.keras.models.load_model(".\\Model\\ChatBotModel_V1.hdf5")

HOST = '0.0.0.0'
PORT = 8081

app = Flask(__name__)
app.debug = True
CORS(app)


@app.route('/chatServer/response', methods=['GET', 'POST'])
def response():
    req = request.get_json()
    text = req["message"]
    text_ready = datasetPreprocessor.preprocess_text_to_predict(text)
    probabilities = chatModel.predict(text_ready)[0]
    prediction = np.argmax(probabilities)

    if probabilities[prediction] <= 0.5:  # pour etre sur de la reponse a plus de 50% au moins
        return jsonify({'response': reponse_not_known})
    else:
        return jsonify({'response': list_responses[prediction]})


@app.errorhandler(500)
def internal_error(error):
    return "500 error"


@app.errorhandler(404)
def not_found(error):
    return "404 error", 404


if __name__ == '__main__':
    # lancement du serveur
    app.run(host=HOST,
            debug=True,  # automatic reloading enabled
            port=PORT)
