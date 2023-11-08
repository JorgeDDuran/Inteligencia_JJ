import sys
import cv2
import os
import imutils
import numpy as np
from PyQt5.QtWidgets import QApplication, QMainWindow, QLabel, QLineEdit, QPushButton, QVBoxLayout, QWidget, QComboBox, QMessageBox
from model.training import train_model
from model.user_model import User
from model.reconocimiento_model import RecognitionLog
from services.reconocimiento_facial import compare_facial_images, capture_facial_image
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QFont
from PyQt5.QtGui import QPixmap

class RegistrationApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.init_ui()

    def init_ui(self):
        self.setWindowTitle("Reconocimiento Facial")
        self.setGeometry(100, 100, 600, 400)

    # Configura el fondo azul utilizando una hoja de estilo
        style = """
            QMainWindow {
                background-color: #5600A3;
            }
        
            QLabel {
                color: salmon;
            }
       
            QPushButton {
                background-color: #4EB1F9;
                color: black;
                border: 2px solid #4EB1F9;
                border-radius: 5px;
                height: 40px;
            }
            QLineEdit {
                border: 2px solid #3498db;
                border-radius: 5px;
                height: 40px;
            }
        """
        self.setStyleSheet(style)

        central_widget = QWidget(self)
        self.setCentralWidget(central_widget)

        layout = QVBoxLayout()

    # Widgets para el registro
        username_label = QLabel("Nombre Completo")
        self.username_input = QLineEdit()
        edad_label = QLabel("Edad")
        self.edad_input = QLineEdit()
        sanguinidad_label = QLabel("Tipo de sangre")
        self.sanguinidad_input = QLineEdit()
        doc_label = QLabel("Documento")
        self.doc_input = QLineEdit()
        
        

        register_button = QPushButton("Registrar")
        register_button.clicked.connect(self.register_user)



    # Agregar elementos de registro al layout
        layout.addWidget(username_label)
        layout.addWidget(self.username_input)
        layout.addWidget(edad_label)
        layout.addWidget(self.edad_input)
        layout.addWidget(sanguinidad_label)
        layout.addWidget(self.sanguinidad_input)
        layout.addWidget(doc_label)
        layout.addWidget(self.doc_input)
        
        
        
        


    
        authenticate_button = QPushButton("Ingresar")
        authenticate_button.clicked.connect(self.authenticate_user)

        layout.addWidget(register_button)
        layout.addWidget(authenticate_button)

    # Botón "Entrenar Modelo"
        train_model_button = QPushButton("Entrenar Modelo")
        train_model_button.clicked.connect(self.train_model)
        layout.addWidget(train_model_button)  # Agrega el botón "Entrenar Modelo"

        central_widget.setLayout(layout)

        
    def train_model(self):
        data_path = 'fotosreconocimiento'  # Ruta de tus datos de entrenamiento
        model_path = 'modeloLBPHFace.xml'  # Ruta donde se guardará el modelo entrenado

    # Llama a la función de entrenamiento desde model.training
        train_model(data_path, model_path)

    # Muestra un cuadro de diálogo con la información del entrenamiento
        msg = QMessageBox()
        msg.setIcon(QMessageBox.Information)
        msg.setWindowTitle("Entrenamiento Completado")

    # Establece el estilo para el texto del cuadro de diálogo
        msg.setStyleSheet("QLabel{ color: black; }")

        msg.setText("El modelo ha sido entrenado con éxito.")
        msg.setStandardButtons(QMessageBox.Ok)

        result = msg.exec_()  # Muestra el cuadro de diálogo


    

    def register_user(self):
        username = self.username_input.text()
        edad = self.edad_input.text()
        sanguinidad = self.sanguinidad_input.text()
        doc = self.doc_input.text()
        
       

        user = User(username, edad, sanguinidad, doc)  # Pasa el rol como tercer argumento
        self.save_user_data(user)
        self.capture_and_save_facial_image(username, edad,  sanguinidad, doc)
   
    

    def save_user_data(self, user):
        username = user.username
        edad = user.edad
        sanguinidad = user.sanguinidad
        doc = user.doc
        
        user_data_file = "user_data.txt"

        with open(user_data_file, "a") as file:
            file.write(f"Nombre Completo: {username}, Edad: {edad}, Tipo de sangre: {sanguinidad}, Documento: {doc}\n")

    def capture_and_save_facial_image(self, username, edad, sanguinidad, doc):
        personName = username +" , " + edad + "," + sanguinidad + ","+ doc
        dataPath = 'fotosreconocimiento' 
        personPath = dataPath + '/' + personName

        if not os.path.exists(personPath):
	        print('Carpeta creada: ',personPath)
	        os.makedirs(personPath)

        cap = cv2.VideoCapture(0,cv2.CAP_DSHOW)

        faceClassif = cv2.CascadeClassifier(cv2.data.haarcascades+'haarcascade_frontalface_default.xml')
        count = 0
        while True:

            ret, frame = cap.read()
            if ret == False: break
            frame =  imutils.resize(frame, width=640)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            auxFrame = frame.copy()

            faces = faceClassif.detectMultiScale(gray,1.3,5)

            for (x,y,w,h) in faces:
                cv2.rectangle(frame, (x,y),(x+w,y+h),(0,255,0),2)
                rostro = auxFrame[y:y+h,x:x+w]
                rostro = cv2.resize(rostro,(150,150),interpolation=cv2.INTER_CUBIC)
                cv2.imwrite(personPath + '/rostro_{}.jpg'.format(count),rostro)
                count = count + 1
            cv2.imshow('frame',frame)

            k =  cv2.waitKey(1)
            if k == 27 or count >= 100:
                break

        cap.release()
        cv2.destroyAllWindows()

    def save_recognition_log(self, recognition_log):
        username = recognition_log.username
        image_path = recognition_log.image_path
        recognition_log_file = "recognition_log.txt"

        with open(recognition_log_file, "a") as file:
            file.write(f"Usuario: {username}, Ruta de la imagen: {image_path}\n")

    def authenticate_user(self):
        
        
        dataPath = 'fotosreconocimiento'
        imagePaths = os.listdir(dataPath)
        print('imagePaths=',imagePaths)
        cap = cv2.VideoCapture(0,cv2.CAP_DSHOW)

        face_recognizer = cv2.face.LBPHFaceRecognizer_create()
        face_recognizer.read('modeloLBPHFace.xml')
        faceClassif = cv2.CascadeClassifier(cv2.data.haarcascades+'haarcascade_frontalface_default.xml')

        while True:
                ret,frame = cap.read()
                if ret == False: break
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                auxFrame = gray.copy()

                faces = faceClassif.detectMultiScale(gray,1.3,5)

                for (x,y,w,h) in faces:
                    rostro = auxFrame[y:y+h,x:x+w]
                    rostro = cv2.resize(rostro,(150,150),interpolation= cv2.INTER_CUBIC)
                    result = face_recognizer.predict(rostro)

                    cv2.putText(frame,'{}'.format(result),(x,y-5),1,1.3,(255,255,0),1,cv2.LINE_AA)
                    
                    if result[1] < 70:
                        cv2.putText(frame,'{}'.format(imagePaths[result[0]]),(x,y-25),2,1.1,(0,255,0),1,cv2.LINE_AA)
                        cv2.rectangle(frame, (x,y),(x+w,y+h),(0,255,0),2)
                    else:
                        cv2.putText(frame,'Desconocido',(x,y-20),2,0.8,(0,0,255),1,cv2.LINE_AA)
                        cv2.rectangle(frame, (x,y),(x+w,y+h),(0,0,255),2)
                    
                cv2.imshow('frame',frame)
                k = cv2.waitKey(1)
                if k == 27:
                    break

        cap.release()
        cv2.destroyAllWindows()

    def verify_user_credentials(self, username, password):
        # Esta función debería consultar una base de datos o archivo para verificar las credenciales del usuario.
        # En este ejemplo, solo se simula la verificación.
        return True

    def get_recognition_log_by_username(self, username):
        # Esta función debería consultar una base de datos o archivo para obtener el registro de reconocimiento del usuario.
        # En este ejemplo, solo se simula la recuperación.
        image_path = "/captured_images"  # Reemplaza con la ubicación real de la imagen
        return RecognitionLog(username, image_path)

    def show_authentication_result(self, message):
        msg = QMessageBox()
        msg.setWindowTitle("Resultado de la autenticación")

        if message == "Autenticación exitosa":
            msg.setIcon(QMessageBox.Information)
            msg.setText("¡Autenticación exitosa! El usuario ha sido reconocido.")
        else:
            msg.setIcon(QMessageBox.Critical)
            msg.setText("Autenticación fallida. Usuario desconocido o credenciales incorrectas.")

        msg.exec_()


    def display_camera_frame(self, frame):
        # Muestra el frame de la cámara en la interfaz gráfica (por ejemplo, en un QLabel)
        pass


if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = RegistrationApp()
    window.show()
    sys.exit(app.exec_())