"""
Módulo principal do programa, responsável por chamar as funções de formatação
de imagens e processamento do tabuleiro de xadrez para calibração da câmera.
"""

import os
from src.images import format_images

CHESSBOARD_SIZE = (9, 9) # Para o tabuleiro 10x10
SQUARE_SIZE_MM = 25.0 # Tamanho do quadrado em milímetros
IMAGES_PATHS = 'images/dataset' # Diretório das imagens de calibração

def main():

    if not os.path.exists(IMAGES_PATHS):
        print("[ERRO] A pasta {} não existe.".format(IMAGES_PATHS))
        exit()

    image_files = [
        os.path.join(IMAGES_PATHS, f) for f in os.listdir(IMAGES_PATHS) if f.lower().endswith(
            ('.jpg', '.jpeg', '.png', '.bmp') # extensões válidas
        )
    ]

    if not image_files:
        print("[ERRO] Nenhuma imagem encontrada na pasta {}.".format(IMAGES_PATHS))
        exit()

    formatted_images, img_size = format_images(image_files)

    if not formatted_images:
        print("ERRO: Nenhuma imagem pôde ser formatada.")
        exit()

if __name__ == "__main__":
    main()