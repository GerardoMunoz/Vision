import cv2
import numpy as np

# Seleccionar el diccionario (4x4_50)
aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
parameters = cv2.aruco.DetectorParameters()

# Crear detector nuevo estilo OpenCV >= 4.7
detector = cv2.aruco.ArucoDetector(aruco_dict, parameters)

# Abrir webcam
cap = cv2.VideoCapture(1)

print("Presiona 'q' para salir...")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Detectar marcadores
    corners, ids, rejected = detector.detectMarkers(frame)

    # Dibujar marcadores si existen
    if ids is not None:
        cv2.aruco.drawDetectedMarkers(frame, corners, ids)
        print("Detectados: corners, ids",   ids)



    cv2.imshow("Aruco 4x4 Detector", frame)


    # Salir con 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    if cv2.getWindowProperty("Aruco 4x4 Detector", cv2.WND_PROP_VISIBLE) < 1:
        print("Ventana cerrada. Saliendo...")
        break


cap.release()
cv2.destroyAllWindows()
