from voice_recorder import voiceRecorder
import text_preprocessor

MyRecorder = voiceRecorder()
# output = MyRecorder.record_audio()
output = "bonjour. Je m'appelle frederic et je suis nouveau a l'ensa"
print(text_preprocessor.preprocess(output))
