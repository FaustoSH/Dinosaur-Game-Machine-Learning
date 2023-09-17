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
import matplotlib.pyplot as plt


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
        self.driver.maximize_window()
        self.action_space = spaces.Discrete(2)  # Salto o no hacer nada
        self.observation_space = spaces.Box(low=0, high=255, shape=(1, 350, 550), dtype=np.uint8)
        self.ejecuciones=0
        self.tiempoInicial=time.time()
        self.puntos=0

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
        y_start, y_end, x_start, x_end = 270, 620, 100, 650 #Coordenadas de detección de obstáculos

        # Recortar la región de interés
        cropped_image = image[y_start:y_end, x_start:x_end]

        # Convertir a escala de grises
        grayscale_image = cv2.cvtColor(cropped_image, cv2.COLOR_BGR2GRAY)

        # Convertir la imagen en un tensor
        image_tensor = torch.tensor(grayscale_image, dtype=torch.uint8) #imagen de detección de obstáculos

        # plt.figure(figsize=(10, 5))
        # plt.imshow(image_tensor)
        # plt.show()

        return image_tensor

    def get_game_state(self):
        screenshot = self.driver.get_screenshot_as_png()
        image = cv2.imdecode(np.frombuffer(screenshot, np.uint8), cv2.IMREAD_COLOR)
        image_tensor = self.preprocess_image(image)


        combined_tensor = np.stack([image_tensor], axis=0)
        return combined_tensor

    def step(self, action):
        action_map={
            0:Keys.SPACE,
            1:'nothing'
        }

        action = int(action)

        if action!=1:
            self.body_element.send_keys(action_map[action])
                    
        done = False
        
        reward = 1 

        self.puntos+=1

        if self.hasDied():
            self.body_element.send_keys(Keys.SPACE)
            done = True
            
        combined_tensor= self.get_game_state()
            
        return combined_tensor, reward, done, False,  {}
    
    def reset(self, seed=None):
        logging.info("Ejecucion numero "+str(self.ejecuciones)+" terminada con "+str(self.puntos)+" puntos")
        self.ejecuciones+=1
        self.puntos=0
        self.driver.refresh()
        time.sleep(1)

        self.tiempoInicial=time.time()

        #Desactivamos las nubes
        self.driver.execute_script("Runner.instance_.horizon.addCloud=function(){}; Runner.instance_.horizon.clouds=[]")

        self.body_element = self.driver.find_element(By.CSS_SELECTOR, 'body')
        self.body_element.send_keys(Keys.ARROW_UP)
        time.sleep(3)
        combined_tensor= self.get_game_state()  # Descomposición de la tupla para obtener solo la observación
        return combined_tensor.astype(np.uint8), {}

    
    def hasDied(self):
        dino=self.driver.execute_script("return Runner.instance_.crashed;")
        return dino
                

    def render(self, mode='human'):
        pass

    def close(self):
        self.driver.quit()

# Creación de la instancia del entorno
env = DinoEnv()
#check_env(env)

buffer_size = 100000  
# Creación del modelo
model = DQN("MlpPolicy", env, verbose=1, buffer_size=buffer_size)

logging.info("Comenzando entrenamiento")
# Entrenamiento
model.learn(total_timesteps=600000) 

# Guardar el modelo
model.save("dino_model")
logging.info("Modelo guardado como dino_model.zip")

# Cargar el modelo (opcional, solo para demostrar)
loaded_model = DQN.load("dino_model", env=env)

logging.info("Comenzando juego")
observation, arrayVacio = env.reset() # Inicialización de la observación
while True:
    action, _states = model.predict(observation)
    observation, reward, done, info, arrayVacio = env.step(action)
    if done:
        logging.info("Partida terminada")
        observation, arrayVacio = env.reset()

