import speech_recognition as sr
from speech_recognition import Recognizer
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer


def Merge(v, Dict):
    return (Dict.update(v))

Dict = {}

recognizer: Recognizer = sr.Recognizer()
with sr.Microphone() as source:
    print('Clearing background noise...')
    recognizer.adjust_for_ambient_noise(source, duration=0.2)
    print('Waiting for your message...')
    recordedaudio = recognizer.listen(source)
    print('Done recording..')

try:
    print('Printing the message..')
    texts = recognizer.recognize_google(recordedaudio, language='en-US')
    print('Your message:{}'.format(texts))
except Exception as ex:
    print(ex)

# Sentiment analysis
Sentence = [str(texts )]
analyser = SentimentIntensityAnalyzer()
for i in Sentence:
    v = analyser.polarity_scores(i)


dict = {}
#This return None
Merge(v, Dict)

# changes made in dict2
#print(Dict)

max_value = max(Dict.values())

max_key = max(Dict, key=Dict.get)
print(max_key)