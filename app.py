from flask import Flask, jsonify, render_template
from flask_cors import CORS
import speech_recognition as sr
import time
import pygame
import os
from transformers import pipeline

app = Flask(__name__)
CORS(app)  

pygame.mixer.init()

# Load the emotion detection model
emotion_pipeline = pipeline("text-classification", model="bhadresh-savani/distilbert-base-uncased-emotion")

def analyze_emotion(text):
    """Analyze text to detect emotion."""
    results = emotion_pipeline(text)
    emotion = max(results, key=lambda x: x['score'])
    return emotion['label'].lower()

def play_music(emotion):
    """Play music based on detected emotion."""
    music_files = {
        "joy": "happy_music.mp3",
        "sadness": "sad_music.mp3",
        "anger": "angry_music.mp3",
        "fear": "fear_music.mp3",
        "neutral": None,
        "disgust": None,
        "surprise": None,
    }

    
    pygame.mixer.music.stop()

   
    music_file = music_files.get(emotion)
    if music_file and os.path.exists(music_file):
        pygame.mixer.music.load(music_file)
        pygame.mixer.music.play(-1)  
        return True
    return False


@app.route('/')
def index():
    return render_template('index.html')

#API
@app.route('/start-detection', methods=['GET'])
def start_detection():
    recognizer = sr.Recognizer()

    with sr.Microphone() as source:
        recognizer.adjust_for_ambient_noise(source)
        audio_data = []

        try:
            start_time = time.time()
            while time.time() - start_time < 10:
                audio_chunk = recognizer.listen(source, timeout=5)
                audio_data.append(audio_chunk)

            # recorded audio
            combined_audio = sr.AudioData(b''.join([chunk.get_raw_data() for chunk in audio_data]), audio_data[0].sample_rate, audio_data[0].sample_width)

            # Convert speech to text
            text = recognizer.recognize_google(combined_audio)
            emotion = analyze_emotion(text)
            music_played = play_music(emotion)

            if music_played:
                return jsonify({"success": True, "emotion": emotion})
            else:
                return jsonify({"success": False, "message": f"No music assigned for {emotion} emotion."})

        except sr.UnknownValueError:
            return jsonify({"success": False, "message": "Could not understand audio."})
        except sr.RequestError as e:
            return jsonify({"success": False, "message": f"Speech recognition request failed: {e}"})

# API to stop music
@app.route('/stop-music', methods=['GET'])
def stop_music():
    pygame.mixer.music.stop()
    return jsonify({"success": True, "message": "Music stopped."})

# Run Flask app
if __name__ == '__main__':
    app.run(debug=True)