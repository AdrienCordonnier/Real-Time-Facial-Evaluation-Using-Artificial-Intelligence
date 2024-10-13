from calculate_beauty_score import *
import cv2

# Chargement du modèle de détection de visage pré-entrainé
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

score = 0
change_com = False
last_comment = "No comment"

# Initialisation de la caméra
cap = cv2.VideoCapture(0)

while True:
    # Capture d'une image depuis la caméra
    ret, frame = cap.read()
    
    # Conversion de l'image en niveaux de gris
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Détection des visages dans l'image
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    
    # Dessiner un cadre autour de chaque visage détecté et enregistrer le visage
    for (x, y, w, h) in faces:
        # Modifier les dimensions du rectangle pour le rendre plus grand
        enlarged_w = int(w * 1.4)  # increase width by 50%
        enlarged_h = int(h * 1.5)  # increase height by 50%
        x = int(x * 0.9)   # Approximated correction to have a full face capture
        y = int(y * 0.7)
        
        # Dessiner un cadre autour du visage
        cv2.rectangle(frame, (x, y), (x+enlarged_w, y+enlarged_h), (255, 0, 0), 2)
        
        # Récupérer le visage en tant qu'image
        face = frame[y:y+enlarged_h, x:x+enlarged_w]
        
        key = cv2.waitKey(1)
        if key == ord('s'):
            cv2.imwrite('face_capture.jpg', face)  # Enregistrer le visage capturé
            print("Image du visage capturée !")
            score  = calculate_score("face_capture.jpg")
            change_com = True
            break

    if score==0:
        text = "Press 's' to capture" + '\n' + "Press 'q' to quit"
    else:
        if change_com:
            com = random_comment(int(score), last_comment)
            last_comment = com
            change_com = False
        text = "I give your face a mark of " + str(round(score, 2)) + '/5' + '\n' + com + '\n' + "Press 'q' to quit"

    # Écrire du texte dans le coin supérieur gauche en rouge et en gras
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.5
    font_thickness = 1
    text_color = (0, 0, 0)  # Black color

    # Dessiner le texte sur deux lignes
    text_lines = text.split('\n')
    for i, line in enumerate(text_lines):
        cv2.putText(frame, line, (10, 30 + i * 20), font, font_scale, text_color, font_thickness, cv2.LINE_AA)
    
    # Affichage de l'image avec les cadres autour des visages
    cv2.imshow('Face Detection', frame)
    
    # Attendre une touche et quitter si la touche 'q' est pressée
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Libération de la capture vidéo et fermeture de la fenêtre
cap.release()
cv2.destroyAllWindows()
