import motmetrics as mm
import numpy as np
import os

def load_mot_file(file_path):
    """Carga un archivo MOT ignorando líneas vacías"""
    if not os.path.exists(file_path):
        print(f"ERROR: No encuentro el archivo {file_path}")
        return {}
    
    # Cargar datos (Frame, ID, X, Y, W, H, Conf, ...)
    try:
        raw_data = np.loadtxt(file_path, delimiter=',')
    except Exception as e:
        print(f"Error leyendo {file_path}: {e}")
        return {}

    frame_dict = {}
    for row in raw_data:
        frame = int(row[0])
        obj_id = int(row[1])
        # Coordenadas de la caja
        x, y, w, h = row[2:6]
        
        if frame not in frame_dict:
            frame_dict[frame] = []
        frame_dict[frame].append({'id': obj_id, 'box': [x, y, w, h]})
    return frame_dict

def evaluar(gt_path, pred_path):
    print(f"--- EVALUANDO SISTEMA COMPLETO ---")
    print(f"GT (Realidad):   {gt_path}")
    print(f"Pred (Tu Flujo): {pred_path}")

    gt = load_mot_file(gt_path)
    ts = load_mot_file(pred_path) # ts = test set (tu predicción)

    if not gt or not ts:
        print("Faltan datos, abortando.")
        return

    acc = mm.MOTAccumulator(auto_id=True)
    
    # Obtener todos los frames unicos
    all_frames = sorted(set(gt.keys()) | set(ts.keys()))
    print(f"Procesando {len(all_frames)} frames...")

    for frame in all_frames:
        gt_objs = gt.get(frame, [])
        ts_objs = ts.get(frame, [])

        gt_ids = [o['id'] for o in gt_objs]
        gt_boxes = [o['box'] for o in gt_objs]
        
        ts_ids = [o['id'] for o in ts_objs]
        ts_boxes = [o['box'] for o in ts_objs]

        # Calcular distancia (IoU) entre cajas
        if gt_boxes and ts_boxes:
            dists = mm.distances.iou_matrix(gt_boxes, ts_boxes, max_iou=0.5)
        else:
            dists = []

        acc.update(gt_ids, ts_ids, dists)

    # Calcular métricas
    mh = mm.metrics.create()
    summary = mh.compute(acc, metrics=['num_frames', 'mota', 'idf1', 'mostly_tracked', 'mostly_lost', 'num_switches'], name='Mi_Modelo')

    # Imprimir tabla bonita
    print("\n" + "="*60)
    print("EVALUACIÓN DE TRACKING")
    print("="*60)
    str_summary = mm.io.render_summary(
        summary,
        formatters=mh.formatters,
        namemap={
            'mota': 'MOTA (Precisión Global)',
            'idf1': 'IDF1 (Calidad Identidad)',
            'mostly_tracked': 'MT (Totalmente Rastreados)',
            'mostly_lost': 'ML (Perdidos)',
            'num_switches': 'ID Sw (Cambios de ID erróneos)'
        }
    )
    print(str_summary)
    print("="*60)

# --- CONFIGURACIÓN ---
# PON AQUÍ LAS RUTAS REALES
ruta_gt_perfecto = r"D:\kaggle_cache\datasets\ayushspai\sportsmot\versions\1\sportsmot_publish\dataset\train\v_1yHWGw8DH4A_c047\gt\gt.txt" 
ruta_tu_resultado = r"C:\Users\Soriano\OneDrive\Documentos\Sport_analytics\code\gta-link - TFG\DeepEIoU_Bundesliga_Split+Connect_eps0.6_minSamples10_K3_mergeDist0.4_spatial1.0\video1.txt"
#ruta_tu_resultado = r"C:\Users\Soriano\OneDrive\Documentos\football_analysis\output_videos\mot_players.txt" tracker del repo de abdullah
#ruta_tu_resultado = r"C:\Users\Soriano\OneDrive\Documentos\Bundesliga\bundesliga reid osnet con entrenado con dataset amateur\video1\gt\gt.txt" = DeepEIoU + Osnet + GTA-Link = r"C:\Users\Soriano\OneDrive\Documentos\Sport_analytics\code\gta-link - TFG\DeepEIoU_Bundesliga_Split+Connect_eps0.6_minSamples10_K3_mergeDist0.4_spatial1.0\video1.txt" son lo mismo

if __name__ == "__main__":
    evaluar(ruta_gt_perfecto, ruta_tu_resultado)