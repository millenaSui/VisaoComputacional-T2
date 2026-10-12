"""
Módulo principal do programa, responsável por chamar as funções de formatação
de imagens e processamento do tabuleiro de xadrez para calibração da câmera.
"""

import os
from src.images import (
    format_images,
    find_chessboard_corners,
    calibrate_camera
)

CHESSBOARD_SIZE = (9, 9) # Para o tabuleiro 10x10
SQUARE_SIZE_MM = 25.0 # Tamanho do quadrado em milímetros
IMAGES_PATHS = 'images/' # diretório das imagens

def main():

    dataset_path = os.path.join(IMAGES_PATHS, "dataset")
    if not os.path.exists(dataset_path):
        print("[ERRO] A pasta {} não existe.".format(dataset_path))
        exit()

    image_files = [
        os.path.join(dataset_path, f) for f in os.listdir(dataset_path) if f.lower().endswith(
            ('.jpg', '.jpeg', '.png', '.bmp') # extensões válidas
        )
    ]

    if not image_files:
        print("[ERRO] Nenhuma imagem encontrada na pasta {}.".format(dataset_path))
        exit()

    formatted_images, img_size = format_images(image_files)

    if not formatted_images:
        print("ERRO: Nenhuma imagem pôde ser formatada.")
        exit()

    corners_path = os.path.join(IMAGES_PATHS, "corners")
    # localiza cantos do tabuleiro em imagens
    imgpoints, valid_images = find_chessboard_corners(
        formatted_images, CHESSBOARD_SIZE, corners_path
    )

    if not imgpoints:
        print("[ERRO] Nenhum canto de tabuleiro foi localizado nas imagens.")
        exit()

    # calibração de câmera e cálculo do erro reprojeção
    calibrate_camera(
        imgpoints, CHESSBOARD_SIZE, SQUARE_SIZE_MM, img_size
    )

if __name__ == "__main__":
    main()