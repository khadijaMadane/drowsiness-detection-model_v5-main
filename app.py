import cv2
import numpy as np
import streamlit as st
import pygame
from tensorflow.keras.models import load_model

# Charger le modèle entraîné
model = load_model('Model.h5')

# Charger les classificateurs en cascade
face_cascade = cv2.CascadeClassifier('haarcascade/haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('haarcascade/haarcascade_eye.xml')

# Configuration de la capture vidéo
cap = cv2.VideoCapture(0)

# Initialisation des variables
Score = 0
alert_displayed = False

# Application Streamlit
st.title('Détection de Somnolence en Temps Réel')

# Placeholder pour l'image finale
output_frame = st.empty()

# Add video stream to the sidebar
video_placeholder = st.sidebar.empty()
video_placeholder.image([], channels='BGR', use_column_width=True)

# Add buttons to the sidebar
with st.sidebar:
    st.markdown('## Controls')
    st.write('<style>div.row-widget.stRadio > div{flex-direction:row;}</style>', unsafe_allow_html=True)
    start_button = st.button('Démarrer', key='start_button', help='Cliquer pour démarrer la détection')
    stop_button = st.button('Arrêter', key='stop_button', help='Cliquer pour arrêter la détection')

# Charger le son de l'alarme
pygame.mixer.init()
alarm_sound = pygame.mixer.Sound('Réveillez-vousréveillez-vous.ogg')
if start_button:
    while True:
        ret, frame = cap.read()
        height, width = frame.shape[0:2]
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5)
        eyes = eye_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=1)
        
        cv2.rectangle(frame, (0, height-50), (200, height), (0, 0, 0), thickness=cv2.FILLED)
        
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, pt1=(x, y), pt2=(x+w, y+h), color=(255, 0, 0), thickness=3)
            
        for (ex, ey, ew, eh) in eyes:
            eye = frame[ey:ey+eh, ex:ex+ew]
            eye = cv2.resize(eye, (80, 80))
            eye = eye / 255
            eye = eye.reshape(80, 80, 3)
            eye = np.expand_dims(eye, axis=0)
            
            prediction = model.predict(eye)
            
            if prediction[0][0] > 0.30: 
                cv2.putText(frame, 'fermés', (10, height-20), fontFace=cv2.FONT_HERSHEY_COMPLEX_SMALL, fontScale=1, color=(255, 255, 255),
                            thickness=1, lineType=cv2.LINE_AA)
                cv2.putText(frame, 'Score: ' + str(Score), (100, height-20), fontFace=cv2.FONT_HERSHEY_COMPLEX_SMALL, fontScale=1, color=(255, 255, 255),
                            thickness=1, lineType=cv2.LINE_AA)
                Score += 1
                if Score > 15 and not alert_displayed:
                    st.warning("Les yeux sont fermés !")
                    # Jouer l'alarme
                    if not pygame.mixer.get_busy():
                        alarm_sound.play()
                    alert_displayed = True
                
            elif prediction[0][1] > 0.90:  
                cv2.putText(frame, 'ouverts', (10, height-20), fontFace=cv2.FONT_HERSHEY_COMPLEX_SMALL, fontScale=1, color=(255, 255, 255),
                            thickness=1, lineType=cv2.LINE_AA)      
                cv2.putText(frame, 'Score: ' + str(Score), (100, height-20), fontFace=cv2.FONT_HERSHEY_COMPLEX_SMALL, fontScale=1, color=(255, 255, 255),
                            thickness=1, lineType=cv2.LINE_AA)
                Score -= 2
                if Score < 0:
                    Score = 0
                if alert_displayed:
                    st.warning("")
                    # Arrêter l'alarme
                    alarm_sound.stop()
                    alert_displayed = False
        
        # Mettre à jour l'image de sortie avec le frame traité
        output_frame.image(frame, channels='BGR', use_column_width=True)
        
        if stop_button:
            break
            
    cap.release()
    cv2.destroyAllWindows()
