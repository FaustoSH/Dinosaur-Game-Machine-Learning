from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.common.exceptions import WebDriverException
import time
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import cv2

driver = webdriver.Chrome()

try:
    driver.get('chrome://dino')
except WebDriverException:
    pass

time.sleep(1)

# Encuentra el elemento body usando el selector CSS
body_element = driver.find_element(By.CSS_SELECTOR, 'body')

# Envía la tecla ARROW_UP para comenzar el juego
body_element.send_keys(Keys.ARROW_UP)

time.sleep(3)

class DinoNN(nn.Module):
    def __init__(self):
        super(DinoNN, self).__init__()
        self.fc1 = nn.Linear(4, 256)  # Ajusta la dimensión de entrada según tus necesidades
        self.fc2 = nn.Linear(256, 2)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

imagenMostrada=False

def preprocess_image(image):
    global imagenMostrada
    # Definir las coordenadas de recorte (por ejemplo)
    y_start, y_end, x_start, x_end = 350, 420, 150, 250

    # Recortar la región de interés
    cropped_image = image[y_start:y_end, x_start:x_end]

    # cv2.rectangle(image, (x_start, y_start), (x_end, y_end), (0, 0, 255), 2)

    # # Mostrar la imagen con el rectángulo rojo
    # if not imagenMostrada:
    #     cv2.imshow('Region to Crop', image)
    #     cv2.waitKey(0)
    #     cv2.destroyAllWindows()
    #     imagenMostrada=True
    
    # Cambiar el tamaño a una resolución más baja
    resized_image = cv2.resize(cropped_image, (80, 80))

    # Convertir a escala de grises
    grayscale_image = cv2.cvtColor(resized_image, cv2.COLOR_BGR2GRAY)

    # Normalizar los valores de los píxeles al rango [0, 1]
    normalized_image = grayscale_image / 255.0

    # Convertir la imagen en un tensor
    image_tensor = torch.tensor(normalized_image)

    object_detected = torch.any(image_tensor < 0.95) # Puedes ajustar este umbral según sea necesario

    return image_tensor, object_detected

def get_game_state(browser):
    # Tomar una captura de pantalla
    screenshot = browser.get_screenshot_as_png()
    image = cv2.imdecode(np.frombuffer(screenshot, np.uint8), cv2.IMREAD_COLOR)
    
    # Procesar la imagen (incluyendo el recorte y la visualización)
    processed_image, object_detected = preprocess_image(image)

    return processed_image, object_detected


def decide_action(action_predictions):
    # Aquí puedes implementar la lógica para decidir qué acción tomar
    # ...
    return action

def perform_action(driver, action):
    # Aquí puedes implementar cómo realizar la acción en el juego (por ejemplo, saltar)
    # ...
    pass

model = DinoNN()

# Bucle de juego
while True:
    image, object_detected = get_game_state(driver)
    if object_detected:
        body_element.send_keys(Keys.ARROW_UP)

    

input("Presiona Enter para cerrar el navegador...")

    #action_predictions = model(game_state)
    # action = decide_action(action_predictions)
    # perform_action(driver, action)

    # Aquí puedes agregar el código para entrenar la red en el juego en tiempo real, si lo deseas
    # ...
