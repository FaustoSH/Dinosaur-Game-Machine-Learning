import re
import csv

# Leer el texto desde un archivo .txt en la misma carpeta
with open("log.txt", "r", encoding="utf-8") as archivo:
    texto = archivo.read()

# Separamos las líneas del texto
lineas = texto.strip().split("\n")

# Inicializamos una lista vacía para almacenar los datos extraídos
datos = []

# Añadimos la cabecera de las columnas
datos.append(["Numero de Ejecucion", "Puntos"])

# Usamos expresiones regulares para extraer los números de cada línea
patron = re.compile(r"Ejecucion numero (\d+) terminada con (\d+) puntos")

for linea in lineas:
    match = patron.match(linea)
    if match:
        numero_ejecucion, numero_choques = match.groups()
        datos.append([numero_ejecucion, numero_choques])

# Escribimos los datos en un archivo CSV
with open("ejecuciones.csv", "w", newline="", encoding="utf-8") as csvfile:
    writer = csv.writer(csvfile)
    writer.writerows(datos)

print("Archivo CSV creado con éxito.")
