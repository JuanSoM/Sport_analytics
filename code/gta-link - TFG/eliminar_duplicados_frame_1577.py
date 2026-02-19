#!/usr/bin/env python3
import argparse
import os

def quitar_duplicados_frame_1577(input_path: str, output_path: str):
    seen = set()
    salida = []
    with open(input_path, 'r', encoding='utf-8') as f:
        for line in f:
            linea = line.rstrip("\n")
            if not linea:
                continue
            partes = linea.split(',')
            try:
                frame = int(float(partes[0]))
            except (ValueError, IndexError):
                salida.append(linea)
                continue
            if frame == 1577:
                # uso de la línea completa como firma para duplicados
                if linea in seen:
                    continue
                seen.add(linea)
            # añadir línea al resultado
            salida.append(linea)

    # escribir salida
    with open(output_path, 'w', encoding='utf-8') as f:
        for l in salida:
            f.write(l + "\n")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Elimina líneas duplicadas exactas en el frame 1577.')
    parser.add_argument('input_file', help='Archivo de entrada MOT')
    parser.add_argument('output_file', nargs='?',
                        help='Archivo de salida; por defecto añade _1577clean')
    args = parser.parse_args()

    entrada = args.input_file
    if args.output_file:
        salida = args.output_file
    else:
        base, ext = os.path.splitext(entrada)
        salida = f"{base}_1577clean{ext}"

    if not os.path.isfile(entrada):
        print(f"Error: fichero {entrada} no encontrado.")
        exit(1)

    quitar_duplicados_frame_1577(entrada, salida)
    print(f"Proceso completado. Archivo guardado en: {salida}")
