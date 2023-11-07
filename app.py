import sys
from PyQt5.QtWidgets import QApplication 
from view.view import RegistrationApp
from model.training import train_model

def main():
    app = QApplication(sys.argv)
    window = RegistrationApp()
    window.show()
    train_model('C:/Users/jddur/OneDrive/Desktop/FOTOS RECON', 'modeloLBPHFace.xml')  # Llama a train_model
    sys.exit(app.exec_())


if __name__ == '__main__':
    main()
    
    

