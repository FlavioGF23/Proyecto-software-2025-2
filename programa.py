import cv2

# Cargar clasificador Haar para rostros
face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

# Leer una imagen
img = cv2.imread("foto.jpg")
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Detectar rostros
faces = face_cascade.detectMultiScale(gray, 1.1, 4)

# Dibujar rectángulos
for (x, y, w, h) in faces:
    cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)

cv2.imshow("Deteccion de Rostros", img)
cv2.waitKey(0)
cv2.destroyAllWindows()
