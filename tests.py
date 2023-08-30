from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.common.exceptions import WebDriverException
import time


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

def hasDied():
    global driver
    dino=driver.execute_script("return Runner.instance_.crashed;")
    print(dino)

while True:
    time.sleep(1)
    hasDied()