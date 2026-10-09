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

    print(f"[INFO] Iniciando leitura e conversão de {len(image_files)} imagens para Grayscale...")

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

    print(f"[INFO] Formatação concluída: {len(formatted_images)} imagens convertidas com sucesso.")
    return formatted_images, image_size


def find_chessboard_corners(formatted_images, chessboard_size, show_corners=False):
    """ 
    Localiza cantos internos do tabuleiro de xadrez em cada imagem. 
    
    :param formatted_images: Lista de tuplas (caminho_da_imagem, imagem_grayscale,
    imagem_original_bgr).
    :param chessboard_size: Tupla (colunas, linhas) indicando a quantidade de
    cantos internos. 
    :param show_corners: Booleano indicando se deve exibir a janela com
    cantos desenhados. 
    
    :return imgpoints: Lista contendo os arrays de pontos
    2D para cada imagem válida. 
    :return valid_images: Lista com as tuplas das imagens em que o tabuleiro
    foi detectado com sucesso.
    """

    imgpoints = []
    valid_images = []

    # critério de parada para o refinamento subpixel (30 iterações)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    total_images = len(formatted_images) 
    print(f"[INFO] Iniciando detecção dos cantos ({chessboard_size} x {chessboard_size[1]}) em {total_images} imagens...")

    for fname, gray, img_bgr in formatted_images:
        found, corners = cv2.findChessboardCorners(gray, chessboard_size, None)
        if not found:
            print(f"[ERRO] Tabuleiro não encontrado na imagem: '{fname}'. Ignorando...")
            continue

        # refinamento subpixel dos cantos detectados
        corners_subpix = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
        imgpoints.append(corners_subpix)
        valid_images.append((fname, gray, img_bgr))

        # exibição opcional
        if show_corners:
            img_copy = img_bgr.copy()
            cv2.drawChessboardCorners(img_copy, chessboard_size, corners_subpix, found)
            cv2.imshow("Cantos do Tabuleiro Detectados", img_copy)
            cv2.waitKey(500)

    if show_corners:
        cv2.destroyAllWindows()

    print(f"[INFO] Detecção concluída: Tabuleiro localizado em {len(imgpoints)}/{total_images} imagens.")
    return imgpoints, valid_images
