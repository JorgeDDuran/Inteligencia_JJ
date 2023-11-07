# user_model.py
class User:
    def __init__(self, username, password, role):
        self.username = username
        self.password = password
        self.role = role  # Nuevo atributo para almacenar el rol

