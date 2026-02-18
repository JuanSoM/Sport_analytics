import cv2
import os
from tqdm import tqdm  # Importar tqdm para la barra de progreso

# Ruta a la carpeta con las imágenes procesadas
img_folder = r"C:\Users\Soriano\OneDrive\Documentos\Bundesliga\bundesliga reid osnet con entrenado con dataset amateur\video1\img1"

# Ruta de salida para el video
output_video = os.path.join(img_folder, "output_video.mp4")

# Obtener la lista de imágenes ordenadas (recorriendo subcarpetas)
images = []
for root, dirs, files in os.walk(img_folder):
    for file in files:
        if file.endswith(".jpg"):
            images.append(os.path.join(root, file))

# Verificar si se encontraron imágenes
if not images:
    print("No se encontraron imágenes en la carpeta.")
    exit()

# Ordenar las imágenes por nombre para asegurar el orden correcto
images = sorted(images)

# Leer la primera imagen para obtener las dimensiones
first_image = cv2.imread(images[0])
height, width, _ = first_image.shape

# Configurar el video (30 FPS)
fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec para MP4
video = cv2.VideoWriter(output_video, fourcc, 30, (width, height))

# Usar tqdm para mostrar la barra de carga mientras agregamos las imágenes al video
for img_path in tqdm(images, desc="Creando video", ncols=100):
    frame = cv2.imread(img_path)
    video.write(frame)

# Liberar el video
video.release()
print(f"Video guardado en: {output_video}")
