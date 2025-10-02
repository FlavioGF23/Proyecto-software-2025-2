import cv2
import os

def detectar_rostros(ruta_imagen):
    # Si la imagen no existe, lanzamos error
    if not os.path.exists(ruta_imagen):
        raise FileNotFoundError(f"No se encontró la imagen: {ruta_imagen}")

    # Cargar clasificador Haar
    face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

    # Leer imagen
    img = cv2.imread(ruta_imagen)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Detectar rostros
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)

    return faces
