"""
Módulo de processamento de imagens para calibração de câmera usando OpenCV.
Este módulo contém funções para:
  - Carregar e formatar imagens (conversão para grayscale).
"""

import cv2

def format_images(image_files):
    """
    Função responsável exclusivamente pelo carregamento
    e conversão das imagens em grayscale.

    :param image_files: Lista com os caminhos dos arquivos de imagem.

    :return formatted_images: Lista de tuplas contendo (caminho_da_imagem, 
    imagem_grayscale, imagem_original_bgr).
    :return image_size: Tupla (largura, altura) da imagem em pixels.
    """
    formatted_images = []
    image_size = None

    print(f"Iniciando leitura e conversão de {len(image_files)} imagens para Grayscale...")

    for fname in image_files:
        img_bgr = cv2.imread(fname)
        if img_bgr is None:
            print(f"[AVISO] Não foi possível carregar a imagem: '{fname}'. Ignorando...")
            continue

        # conversão para grayscale
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

        if image_size is None:
            image_size = gray.shape[::-1]

        formatted_images.append((fname, gray, img_bgr))

    print(f"[INFO] Formatação concluída: {len(formatted_images)} imagens convertidas com sucesso.\n")
    return formatted_images, image_size
