from voice_recorder import voiceRecorder
import text_preprocessor

MyRecorder = voiceRecorder()
output = MyRecorder.record_audio()
print(text_preprocessor.preprocess(output))
