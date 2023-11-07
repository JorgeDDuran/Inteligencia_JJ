import cv2
import os
import numpy as np

def train_model(dataPath, modelPath):
    peopleList = os.listdir(dataPath)
    print('Lista de personas: ', peopleList)

    labels = []
    facesData = []
    label = 0

    for nameDir in peopleList:
        personPath = os.path.join(dataPath, nameDir)
        print('Leyendo las imágenes')

        for fileName in os.listdir(personPath):
            print('Rostros: ', nameDir + '/' + fileName)
            labels.append(label)
            facesData.append(cv2.imread(os.path.join(personPath, fileName), 0))

        label = label + 1

    face_recognizer = cv2.face.LBPHFaceRecognizer_create()

    print("Entrenando...")
    face_recognizer.train(facesData, np.array(labels))

    face_recognizer.write(modelPath)
    print("Modelo almacenado...")
