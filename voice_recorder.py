import speech_recognition as sr  # recognise speech
import pyttsx3


class voiceRecorder:
    def __init__(self):
        self.recognizer = sr.Recognizer()  # initialise a recogniser
        self.engine = pyttsx3.init()

    def engine_speak(self, text):
        text = str(text)
        self.engine.say(text)
        self.engine.runAndWait()

    # listen for audio and convert it to text:
    def record_audio(self, ask=""):
        with sr.Microphone() as source:  # microphone as source
            if ask:
                self.engine_speak(ask)
            try:
                audio = self.recognizer.listen(source, 15, 150)  # listen for the audio via source
            except sr.WaitTimeoutError as e:
                print("Timeout; {0}".format(e))
            print("Done Listening")
            voice_data = ''
            try:
                voice_data = self.recognizer.recognize_google(audio, language="fr-FR")  # convert audio to text
            except sr.UnknownValueError:  # error: recognizer does not understand
                self.engine_speak('I did not get that')
            except sr.RequestError:
                self.engine_speak('Sorry, the service is down')  # error: recognizer is not connected
            except UnboundLocalError:
                self.engine_speak('Sorry, the Audio is None')

            print(">>", voice_data.lower())  # print what user said
            return voice_data.lower()
