from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.common.exceptions import WebDriverException
import time
import torch
import cv2
import gymnasium as gym
from stable_baselines3 import DQN
from stable_baselines3.common.env_checker import check_env
from gymnasium import spaces
import numpy as np
import torch
import logging

logging.basicConfig(level=logging.INFO,
                    format='%(message)s',
                    handlers=[logging.FileHandler('log.txt', mode='w'), logging.StreamHandler()])


if torch.cuda.is_available():
    device = torch.device("cuda")
    logging.info("GPU is available")
else:
    device = torch.device("cpu")
    logging.info("GPU not available, using CPU")


class DinoEnv(gym.Env):
    def __init__(self):
        super(DinoEnv, self).__init__()
        self.driver = webdriver.Chrome()
        self.action_space = spaces.Discrete(3)  # Salto, agacharse o no hacer nada
        self.observation_space = spaces.Box(low=0, high=1, shape=(2, 80, 80), dtype=np.float32)
        self.points = 0

        try:
            self.driver.get('chrome://dino')
        except WebDriverException:
            pass

        time.sleep(1)
        self.body_element = self.driver.find_element(By.CSS_SELECTOR, 'body')
        self.body_element.send_keys(Keys.ARROW_UP)
        time.sleep(3)

    def preprocess_image(self, image):
        # Definir las coordenadas de recorte (por ejemplo)
        y_start, y_end, x_start, x_end = 200, 435, 150, 1500 #Coordenadas de detección de obstáculos
        y_start1, y_end1, x_start1, x_end1 = 200, 435, 0, 150 #Coordenadas de detección de dinosaurio

        # Recortar la región de interés
        cropped_image = image[y_start:y_end, x_start:x_end]
        cropped_image1 = image[y_start1:y_end1, x_start1:x_end1]

        # cv2.rectangle(image, (x_start, y_start), (x_end, y_end), (0, 0, 255), 2)
        # cv2.rectangle(image, (x_start1, y_start1), (x_end1, y_end1), (0, 0, 255), 2)

        # cv2.imshow('Region to Crop', image)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
        
        # Cambiar el tamaño a una resolución más baja
        resized_image = cv2.resize(cropped_image, (80, 80))
        resized_image1 = cv2.resize(cropped_image1, (80, 80))

        # Convertir a escala de grises
        grayscale_image = cv2.cvtColor(resized_image, cv2.COLOR_BGR2GRAY)
        grayscale_image1 = cv2.cvtColor(resized_image1, cv2.COLOR_BGR2GRAY)

        # Normalizar los valores de los píxeles al rango [0, 1]
        normalized_image = grayscale_image / 255.0
        normalized_image1 = grayscale_image1 / 255.0

        # Convertir la imagen en un tensor
        image_tensor = torch.tensor(normalized_image, dtype=torch.float32) #imagen de detección de obstáculos
        image_tensor1 = torch.tensor(normalized_image1, dtype=torch.float32) #imagen de detección de salto

        return image_tensor, image_tensor1

    def get_game_state(self):
        screenshot = self.driver.get_screenshot_as_png()
        image = cv2.imdecode(np.frombuffer(screenshot, np.uint8), cv2.IMREAD_COLOR)
        image_tensor, image_tensor1 = self.preprocess_image(image)

        combined_tensor = np.stack([image_tensor, image_tensor1], axis=0)
        return combined_tensor

    def step(self, action):
        if action == 0:
            self.body_element.send_keys(Keys.ARROW_UP)  # Saltar
        elif action == 1:
            self.body_element.send_keys(Keys.ARROW_DOWN)  # Agacharse
        
        #No defino la acción 3 que es no hacer nada porque no hace nada xd
            
        done = False
        reward = 1 

        if self.hasDied():
            logging.info("Encontrado final")
            self.body_element.send_keys(Keys.ARROW_UP)
            done = True
            reward = -12
        elif self.dodgingObstable():
            reward = 3
        elif self.isMoving():
            reward -3

            
        combined_tensor= self.get_game_state()
            

        self.points += reward

        return combined_tensor, reward, done, False,  {}
    
    def reset(self, seed=None):
        logging.info("Ha llegado hasta "+str(self.points)+" puntos")
        self.points=0
        self.driver.refresh()
        time.sleep(1)
        self.body_element = self.driver.find_element(By.CSS_SELECTOR, 'body')
        self.body_element.send_keys(Keys.ARROW_UP)
        time.sleep(3)
        combined_tensor= self.get_game_state()  # Descomposición de la tupla para obtener solo la observación
        return combined_tensor.astype(np.float32), {}


    def hasDied(self):
        dino=self.driver.execute_script("return Runner.instance_.crashed;")
        return dino
    
    def dodgingObstable(self):
        distanciaDePrevision = 10

        # Obtener todas las propiedades necesarias en una sola llamada de JavaScript
        script = """
        return {
            trexWidth: Runner.instance_.tRex.config.WIDTH,
            trexWidthDuck: Runner.instance_.tRex.config.WIDTH_DUCK,
            trexDucking: Runner.instance_.tRex.ducking,
            nearestObstacle: Runner.instance_.horizon.obstacles.length > 0 ? Runner.instance_.horizon.obstacles[0] : null
        };
        """

        data = self.driver.execute_script(script)

        # Extraer datos
        trexWidth = data['trexWidth']
        trexWidthDuck = data['trexWidthDuck']
        trexDucking = data['trexDucking']
        nearestObstacle = data['nearestObstacle']

        # Verificar si hay un obstáculo cercano
        if nearestObstacle:
            nearestObstacleWidth = nearestObstacle['width']
            nearestObstacleXPosition = nearestObstacle['xPos']
            effectiveTrexWidth = trexWidthDuck if trexDucking else trexWidth
            isDodging = (nearestObstacleXPosition - nearestObstacleWidth) <= (effectiveTrexWidth + distanciaDePrevision)
        else:
            isDodging = False

        return isDodging
    
    def isMoving(self):
        script = """
        return {
            trexDucking: Runner.instance_.tRex.ducking,
            trexJumping: Runner.instance_.tRex.jumping
        };
        """

        data = self.driver.execute_script(script)

        trexDucking = data["trexDucking"]
        trexJumping = data["trexJumping"]

        if trexDucking or trexJumping:
            return True
        else:
            return False

        

    def render(self, mode='human'):
        pass

    def close(self):
        self.driver.quit()

# Creación de la instancia del entorno
env = DinoEnv()
check_env(env)

buffer_size = 100000  
# Creación del modelo
model = DQN("MlpPolicy", env, verbose=1, buffer_size=buffer_size)

logging.info("Comenzando entrenamiento")
# Entrenamiento
model.learn(total_timesteps=600000)  # Aproximadamente 10 minutos de entrenamiento

logging.info("Comenzando juego")
observation, arrayVacio = env.reset() # Inicialización de la observación
while True:
    action, _states = model.predict(observation)
    observation, reward, done, info, arrayVacio = env.step(action)
    if done:
        logging.info("Partida terminada")
        observation, arrayVacio = env.reset()

