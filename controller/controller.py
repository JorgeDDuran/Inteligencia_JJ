from model.user_model import User
from model.reconocimiento_model import RecognitionLog
from services.reconocimiento_facial import capture_facial_image, compare_facial_images

# Lista para almacenar los registros de reconocimiento
recognition_logs = []

def register_user(username, password):
    # Validar la información del usuario (por ejemplo, verificar si el usuario ya existe)
    
    # Crear una instancia de la clase User con los datos proporcionados
    user = User(username, password)
    
    # Definir la ruta del archivo donde se guardarán los datos de usuario
    user_data_file = "user_data.txt"
    
    # Guardar los datos del usuario en el archivo
    with open(user_data_file, "a") as file:
        file.write(f"Usuario: {user.username}, Contraseña: {user.password}\n")

def authenticate_user(username, password):
    # Define la ruta del archivo donde se almacenan los datos de usuario
    user_data_file = "user_data.txt"
    
    # Verifica si el usuario y la contraseña coinciden con los datos almacenados
    with open(user_data_file, "r") as file:
        for line in file:
            if f"Usuario: {username}, Contraseña: {password}" in line:
                return True  # Usuario autenticado

    return False  # Usuario no autenticado

def register_user_with_facial_recognition(username, password):
    # Valida y guarda la información del usuario en una estructura de datos o archivo
    user = User(username, password)
    # Realiza el registro del usuario (almacenamiento en la base de datos o archivo)

    # Captura la imagen facial
    image_path = capture_facial_image()
    if image_path:
        recognition_log = RecognitionLog(username, image_path)
        # Guarda el registro de reconocimiento (almacenamiento en la base de datos o archivo)
        recognition_logs.append(recognition_log)
    
def authenticate_user_with_facial_recognition(username):
    # Recupera la imagen facial capturada del servicio de reconocimiento
    captured_image_path = capture_facial_image()

    if captured_image_path:
        # Recupera el registro de reconocimiento para el usuario
        recognition_log = get_recognition_log_by_username(username)

        if recognition_log is not None:
            # Compara la imagen facial capturada con la almacenada en el registro de reconocimiento
            if compare_facial_images(captured_image_path, recognition_log.image_path):
                return True  # La autenticación facial es exitosa
    return False  # La autenticación facial falla

def get_recognition_log_by_username(username):
    # Itera a través de la lista de registros y busca el registro correspondiente al nombre de usuario
    for recognition_log in recognition_logs:
        if recognition_log.username == username:
            return recognition_log  # Devuelve el registro si se encuentra

    return None  # Devuelve None si no se encuentra ningún registro para el nombre de usuario
