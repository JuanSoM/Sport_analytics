import cv2
import os
import numpy as np
from tqdm import tqdm

# ==========================================
# CONFIGURACIÓN (Pon tus rutas aquí)
# ==========================================
# Carpeta donde tienes las fotos originales (000001.jpg, 000002.jpg...)
FRAMES_DIR = r"C:\Users\Soriano\OneDrive\Documentos\Bundesliga\bundesliga reid osnet con entrenado con dataset amateur\video1\img1" 

# El archivo .txt que quieres visualizar (tu resultado de GTA-Link o el GT perfecto)
MOT_FILE = r"C:\Users\Soriano\OneDrive\Documentos\Sport_analytics\code\gta-link - TFG\DeepEIoU_Bundesliga_Split+Connect_eps0.6_minSamples10_K3_mergeDist0.4_spatial1.0\video1.txt" 

# Dónde quieres guardar el vídeo resultante y cómo se llamará
OUTPUT_VIDEO = r"C:\Users\Soriano\OneDrive\Documentos\Sport_analytics\code\gta-link - TFG\video_etiquetado_DeepEIoU_miModelo_gta_link.mp4"

# A cuántos FPS quieres que vaya el vídeo (suele ser 25 o 30)
FPS = 25
# ==========================================

def generar_color(id_obj):
    """Genera un color pseudo-aleatorio pero constante para cada ID."""
    np.random.seed(id_obj * 100)
    color = np.random.randint(0, 255, size=3)
    return tuple(int(c) for c in color)

def load_mot_data(txt_path):
    """Lee el archivo MOT y lo agrupa por frames."""
    print(f"Leyendo archivo MOT: {txt_path}")
    data = np.loadtxt(txt_path, delimiter=',')
    
    mot_dict = {}
    for row in data:
        frame = int(row[0])
        obj_id = int(row[1])
        x, y, w, h = map(int, row[2:6])
        
        if frame not in mot_dict:
            mot_dict[frame] = []
        mot_dict[frame].append((obj_id, x, y, w, h))
        
    return mot_dict

def main():
    # 1. Cargar las cajas y los IDs
    mot_data = load_mot_data(MOT_FILE)
    
    # 2. Leer la lista de imágenes de la carpeta
    imagenes = [f for f in os.listdir(FRAMES_DIR) if f.endswith(('.jpg', '.png'))]
    imagenes.sort() # Asegurar que van en orden (000001, 000002...)
    
    if not imagenes:
        print("No se encontraron imágenes en la carpeta.")
        return

    # 3. Preparar el creador de vídeo (VideoWriter)
    primera_img = cv2.imread(os.path.join(FRAMES_DIR, imagenes[0]))
    alto, ancho = primera_img.shape[:2]
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v') # Formato MP4
    video_out = cv2.VideoWriter(OUTPUT_VIDEO, fourcc, FPS, (ancho, alto))
    
    print(f"Generando vídeo en: {OUTPUT_VIDEO}")
    print(f"Resolución: {ancho}x{alto} a {FPS} FPS")
    
    # 4. Bucle principal: Pintar frame a frame
    for frame_idx, img_name in enumerate(tqdm(imagenes)):
        # El frame_idx empieza en 0, pero los MOT suelen empezar en el frame 1
        frame_num = frame_idx + 1 
        
        img_path = os.path.join(FRAMES_DIR, img_name)
        frame_img = cv2.imread(img_path)
        
        # Si hay detecciones para este frame, las pintamos
        if frame_num in mot_data:
            for obj_id, x, y, w, h in mot_data[frame_num]:
                # Calcular coordenadas (asegurando que no se salgan de la pantalla)
                x1, y1 = max(0, x), max(0, y)
                x2, y2 = min(ancho, x + w), min(alto, y + h)
                
                # Obtener un color único para este ID
                color = generar_color(obj_id)
                
                # Pintar la caja (grosor 2)
                cv2.rectangle(frame_img, (x1, y1), (x2, y2), color, 2)
                
                # Pintar el fondo del texto para que se lea mejor
                etiqueta = f"ID: {obj_id}"
                cv2.rectangle(frame_img, (x1, y1 - 20), (x1 + len(etiqueta)*10, y1), color, -1)
                
                # Escribir el ID
                cv2.putText(frame_img, etiqueta, (x1 + 2, y1 - 5), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        
        # Guardar el frame ya pintado en el vídeo
        video_out.write(frame_img)
        
    # Cerrar el archivo de vídeo
    video_out.release()
    print("¡Vídeo generado con éxito!")

if __name__ == "__main__":
    main()