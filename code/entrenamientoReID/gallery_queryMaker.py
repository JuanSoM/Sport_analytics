import os
import cv2
import numpy as np
import shutil
import random
from tqdm import tqdm

# --- CONFIGURACIÓN ---
# Rutas actualizadas según lo que me has pasado
PATH_FRAMES = r"D:\kaggle_cache\datasets\ayushspai\sportsmot\versions\1\sportsmot_publish\dataset\train\v_1yHWGw8DH4A_c047\img1"
PATH_GT = r"D:\kaggle_cache\datasets\ayushspai\sportsmot\versions\1\sportsmot_publish\dataset\train\v_1yHWGw8DH4A_c047\gt\gt.txt"
OUTPUT_DIR = r"C:\Users\Soriano\OneDrive\Documentos\entrenamientoReID\player_crops_TEST_GT_Bundesliga"

# Porcentaje de imágenes que irán a Query (el resto a Gallery)
QUERY_RATIO = 0.5  # 50% para query, 50% para gallery

# Extensión de tus imágenes
IMG_EXT = ".jpg" 
# ---------------------

def create_structure(base_path):
    """Crea las carpetas necesarias limpiando si ya existen"""
    if os.path.exists(base_path):
        print(f"Borrando directorio existente: {base_path}...")
        shutil.rmtree(base_path)
    
    os.makedirs(os.path.join(base_path, "query"), exist_ok=True)
    os.makedirs(os.path.join(base_path, "gallery"), exist_ok=True)
    # Carpeta temporal para guardar todos los crops antes de repartir
    temp_path = os.path.join(base_path, "temp_all")
    os.makedirs(temp_path, exist_ok=True)
    return temp_path

def load_gt(gt_path):
    """Carga el GT y lo agrupa por frames"""
    print(f"Cargando GT desde {gt_path}...")
    try:
        data = np.loadtxt(gt_path, delimiter=',')
    except Exception as e:
        print(f"Error cargando GT: {e}")
        return {}

    gt_by_frame = {}
    for row in data:
        frame = int(row[0])
        pid = int(row[1])
        x, y, w, h = map(int, row[2:6])
        
        # Ignorar IDs negativos
        if pid < 0: continue

        if frame not in gt_by_frame:
            gt_by_frame[frame] = []
        gt_by_frame[frame].append({'id': pid, 'box': [x, y, w, h]})
    
    return gt_by_frame

def crop_images(frames_path, gt_data, temp_output_path):
    print("Recortando jugadores de los frames...")
    
    frames_sorted = sorted(gt_data.keys())
    
    for frame_id in tqdm(frames_sorted):
        # Construir nombre del archivo (formato 000001.jpg)
        img_name = f"{frame_id:06d}{IMG_EXT}"
        img_full_path = os.path.join(frames_path, img_name)
        
        if not os.path.exists(img_full_path):
            img_name = f"{frame_id}{IMG_EXT}"
            img_full_path = os.path.join(frames_path, img_name)
            if not os.path.exists(img_full_path):
                continue

        img = cv2.imread(img_full_path)
        if img is None: continue
        
        height, width, _ = img.shape
        
        for obj in gt_data[frame_id]:
            pid = obj['id']
            x, y, w, h = obj['box']
            
            x1 = max(0, x)
            y1 = max(0, y)
            x2 = min(width, x + w)
            y2 = min(height, y + h)
            
            if x2 <= x1 or y2 <= y1: continue
            
            crop = img[y1:y2, x1:x2]
            
            # --- CAMBIO IMPORTANTE AQUÍ ---
            # Añadimos el prefijo "player_" al nombre de la carpeta
            # Esto hará que las carpetas sean "player_0001", "player_0002", etc.
            id_folder_name = f"player_{pid:04d}"
            id_folder = os.path.join(temp_output_path, id_folder_name)
            os.makedirs(id_folder, exist_ok=True)
            
            save_name = f"player_{pid:04d}_f{frame_id:06d}.jpg"
            cv2.imwrite(os.path.join(id_folder, save_name), crop)

def split_query_gallery(base_path, temp_path):
    print("Repartiendo entre Query y Gallery...")
    
    # Listar todas las carpetas (que ahora se llaman player_XXXX)
    person_ids_folders = os.listdir(temp_path)
    
    for folder_name in tqdm(person_ids_folders):
        pid_path = os.path.join(temp_path, folder_name)
        images = os.listdir(pid_path)
        
        random.shuffle(images)
        num_imgs = len(images)
        
        if num_imgs < 2:
            split_idx = 0 
        else:
            split_idx = int(num_imgs * QUERY_RATIO)
            if split_idx == num_imgs:
                split_idx = num_imgs - 1
        
        query_imgs = images[:split_idx]
        gallery_imgs = images[split_idx:]
        
        # Mover archivos manteniendo el nombre de carpeta "player_XXXX"
        
        # 1. Query
        query_dest = os.path.join(base_path, "query", folder_name)
        os.makedirs(query_dest, exist_ok=True)
        for img in query_imgs:
            shutil.move(os.path.join(pid_path, img), os.path.join(query_dest, img))
            
        # 2. Gallery
        gallery_dest = os.path.join(base_path, "gallery", folder_name)
        os.makedirs(gallery_dest, exist_ok=True)
        for img in gallery_imgs:
            shutil.move(os.path.join(pid_path, img), os.path.join(gallery_dest, img))
            
    shutil.rmtree(temp_path)
    print("¡Hecho! Carpeta temporal eliminada.")

if __name__ == "__main__":
    temp_folder = create_structure(OUTPUT_DIR)
    gt_data = load_gt(PATH_GT)
    
    if len(gt_data) > 0:
        crop_images(PATH_FRAMES, gt_data, temp_folder)
        split_query_gallery(OUTPUT_DIR, temp_folder)
        print(f"\nDataset compatible generado en: {OUTPUT_DIR}")
    else:
        print("No se encontraron datos en el GT.")