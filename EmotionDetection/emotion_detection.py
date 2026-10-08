"""Emotion detection using the Watson NLP embeddable library (EmotionPredict)."""
import json
import requests


def emotion_detector(text_to_analyze):
    """Return the emotion scores and the dominant emotion for the given text.

    For blank entries the server answers with status code 400 and every
    value in the returned dictionary is None.
    """
    url = ('https://sn-watson-emotion.labs.skills.network/v1/'
           'watson.runtime.nlp.v1/NlpService/EmotionPredict')
    headers = {"grpc-metadata-mm-model-id": "emotion_aggregated-workflow_lang_en_stock"}
    myobj = {"raw_document": {"text": text_to_analyze}}
    response = requests.post(url, json=myobj, headers=headers, timeout=30)

    # Error handling for blank entries
    if response.status_code == 400:
        return {
            'anger': None,
            'disgust': None,
            'fear': None,
            'joy': None,
            'sadness': None,
            'dominant_emotion': None
        }

    # Convert the response text into a dictionary
    formatted_response = json.loads(response.text)

    # Extract the required set of emotions and their scores
    emotions = formatted_response['emotionPredictions'][0]['emotion']
    result = {
        'anger': emotions['anger'],
        'disgust': emotions['disgust'],
        'fear': emotions['fear'],
        'joy': emotions['joy'],
        'sadness': emotions['sadness']
    }

    # The dominant emotion is the one with the highest score
    result['dominant_emotion'] = max(result, key=result.get)
    return result
