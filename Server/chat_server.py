from flask import Flask, jsonify, request
from flask_cors import CORS
import tensorflow as tf
from Data import dataset_pre_processor, online_data_service
import numpy as np

# liste des reponses appropriees a chaques questions

reponse_question_0 = "De nombreux dossiers sont a fournir pour s'incrire a l'Ecole Nationale Des SCiences\
                     Appliquees de Marrakech. Vous pouvez en avoir une liste exhaustive en cliquant sur\
                      le lien suivants : \n https://www.ensa.ac.ma/docs/concours/concours_3_2013/pieces_a_fournir.pdf"

reponse_question_1 = "L'annee scolaire 2022-2023 a l'ENSA Marrakech va de septembre 2022 a juin 2023\nLa\
                        periode des vacances scolaires va de .. Janvier 2023 a  .. Fevrier 2023 puis de \
                        .. Juillet 2023 a .. septembre 2023\n Les Emplois du temps pour chaque niveau et \
                     filiere est disponible sur le lien suivant :\n https://www.ensa.ac.ma/emploisDuTemps.php"

reponse_question_2 = "Pour faire votre inscription a l'ENSA Marrakech, Vous devez tout d'abord vous munir\
                     de toutes les pieces requises, puis vous rendre au services des scolarite de l'ENSA,\
                     du lundi au vendredi de 8h30 a 12h30 et de 14h30 a 17h30"

reponse_question_3 = "- L'assurance scolaire d'un étudiant couvre votre enfant en âge de faire des\
                        études universitaires (BTS, université, classe préparatoire, école de commerce, \
                        etc.). Elle leur apporte une couverture pendant leurs études, mais aussi dans le  \
                        cadre de leurs stages.\n Le prix de l'assurance varie chaque année à partir\
                         de 45 dh.\nPas de papiers nécessaire pour avoir l'assurance, il faut s'inscrire\
                          dans une plateforme pour bénéficier des services "

reponse_question_4 = "La vie associative est la participation aux activités d'une association. \
                        Une association est un groupe de personnes mu par des objectifs communs.Etre\
                         étudiant ne doit pas se limiter à aller en cours puis repartir. Mais sans\
                          négliger qu'il faut  se mettre une barrière pour ne pas passer plus de temps \
                          sur le monde associatif que sur celui des études.\nAvoir une vie associative permet\
                           aux etudiants:\nAcquérir la compétence\n\tDevenir Leader\n\tParticiper au\
                            développement\n\tApprendre à vivre en société\n\tValoriser son profil"

reponse_question_5 = "Si la crédibilité accrue d'une association est notamment due au fait que son\
                             existence est la preuve de votre capacité à vous organiser et à structurer\
                              votre action, la création de votre association prend du temps : il vous\
                               faut vous mettre d'accord sur ses buts, rédiger des statuts, élire un \
                               comité. Puis les membres seront régulièrement impliqués, ne serait-ce que\
                                pour les assemblées générales annuelles. L'engagement dans une \
                                association n'est donc pas quelque chose qui se fait à la légère."

reponse_question_6 = "L'actuel Directeur de l'Ecole Nationale des Sciences Appliquées de Marrakech est" \
                     "\n le Professeur Mohamed AIT FDIL. \n L'actuel Directeur ajoint est " \
                     "le Professeur Hassan AYAD"

reponse_question_7 = "La Formation à l'ENSA est dispensée sous forme d’enseignements théoriques et " \
                     "d’activités pratiques organisés en modules et en semestres\nL'ENSA de Marrakech " \
                     "offre des formations initiales en vue d'obtention de diplôme d'ingénieur d'état " \
                     "dans les spécialités suivantes: \n\tSystèmes Electroniques Embarqués et Commande" \
                     " des Systèmes\n\tGénie Industriel et Logistique\n\tGénie Informatique\n\tRéseaux," \
                     " Systèmes & Services Programmables\n\tGénie Cyber-Défense et Systèmes de " \
                     "Télécommunications Embarqués"

reponse_question_8 = "De nombreuses possibilites s'offrent aux etudiants pour trouver un logement. Tout" \
                     "d'abord vous pouvez faire une demande pour avoir un logement dans une des cites" \
                     "universitaires de Marrakech : \n\t Cité Universitaire Amerchich - Dawdiyat " \
                     "(Filles et Garçons) \n\tAnnexe Cité Universitaire, Avenue Abdelkrim Khatabi -" \
                     " Guéliz (Filles).\n Les demandes peuvent se faire sur le site" \
                     " https://www.onousc.ma/\n\t Vous avez egalement la possibilite de former des " \
                     "collocations avec d'autres etudiants de l'ENSA marrakech. Vous pouvez voir ou publier des" \
                     "annonces par exemple sur des groupes facebook dedies : \n" \
                     "https://fr-fr.facebook.com/groups/682100445211979/?mibextid=6NoCDW"

reponse_question_9 = "Bonjour. En quoi puis-je vous etre utile ?"

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
dictionary = dataset = online_data_service.get_data_from_sheet(online_data_service.LexiqueGShetName,
                                                               online_data_service.TabName)
dataset = online_data_service.get_data_from_sheet(online_data_service.GSheetName, online_data_service.TabName)

# faire le preprocess du dataset pour pouvoir les utiliser pour les predictions
print("preprocessing datasets for text encoding...")
datasetPreprocessor = dataset_pre_processor.DatasetPreprocessor(dictionary, dataset, 1)

# charger le model
print("Loading model ...")
chatModel = tf.keras.models.load_model("../Model/ChatBotModel_V1.hdf5")

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
    print(probabilities)
    prediction = np.argmax(probabilities)

    if probabilities[prediction] <= 0.5:  # pour etre sur de la reponse a plus de 50% de probabilite  au moins
        # on va ajouter quand meme le texte a notre dataset, le but etant de pouvoir trouver de nouvelles categories
        dataset.loc[len(dataset.index)] = [text, -1]
        online_data_service.write_data_to_sheet(online_data_service.GSheetName, online_data_service.TabName, dataset)

        return jsonify({'response': reponse_not_known})
    else:
        # on va ajouter la nouvelle prediction a notre dataset en ligne
        dataset.loc[len(dataset.index)] = [text, prediction]
        online_data_service.write_data_to_sheet(online_data_service.GSheetName, online_data_service.TabName, dataset)

        return jsonify({'response': list_responses[prediction]})


@app.route('/chatServer/start', methods=['GET', 'POST'])
def start():
    return jsonify({'response': "Je suis le chatBot EnsaBot.\n Je suis là pour apporter un maximun de reponses"
                                "à vos differentes questions et inquietudes\n Pour quiter la conversation saisissez "
                                "quit"})


@app.errorhandler(500)
def internal_error(error):
    return "500 error"


@app.errorhandler(404)
def not_found(error):
    return "404 error", 404


if __name__ == '__main__':
    # lancement du serveur
    app.run(host=HOST,
            debug=False,  # automatic reloading not enabled
            port=PORT)
