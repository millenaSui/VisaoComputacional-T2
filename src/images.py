"""
Módulo de processamento de imagens para calibração de câmera usando OpenCV.
Este módulo contém funções para:
  - Carregar e formatar imagens (conversão para grayscale).
"""

import os
import cv2
import numpy as np

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


def find_chessboard_corners(formatted_images, chessboard_size, corners_path):
    """ 
    Localiza cantos internos do tabuleiro de xadrez em cada imagem. 
    
    :param formatted_images: Lista de tuplas (caminho_da_imagem, imagem_grayscale,
    imagem_original_bgr).
    :param chessboard_size: Tupla (colunas, linhas) indicando a quantidade de
    cantos internos. 
    :param corners_path: Caminho para o diretório onde as imagens com os cantos detectados serão salvas.
    :param show_corners: Booleano indicando se deve exibir a janela com
    cantos desenhados. 
    
    :return imgpoints: Lista contendo os arrays de pontos
    2D para cada imagem válida. 
    :return valid_images: Lista com as tuplas das imagens em que o tabuleiro
    foi detectado com sucesso.
    """

    imgpoints = []
    valid_images = []
    
    # cria diretório para salvar imagens com os cantos detectados
    os.makedirs(corners_path, exist_ok=True)

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

        # salva a imagem com cantos detectados
        img_copy = img_bgr.copy()
        cv2.drawChessboardCorners(img_copy, chessboard_size, corners_subpix, found)
        
        base_name = os.path.basename(fname)
        save_path = os.path.join(corners_path, base_name)
        cv2.imwrite(save_path, img_copy)

    print(f"[INFO] Detecção concluída: Tabuleiro localizado em {len(imgpoints)}/{total_images} imagens.")
    print(f"[INFO] As imagens com os cantos detectados foram salvas em '{corners_path}'.")
    return imgpoints, valid_images


def generate_object_points(chessboard_size, square_size_mm):
    """
    Gera as coordenadas dos cantos internos do tabuleiro (plano em Z = 0).

    :param chessboard_size: Tupla (colunas, linhas) indicando os cantos internos.
    :param square_size_mm: Tamanho do quadrado em milímetros.

    :return objp: Array numpy (N, 3) com as coordenadas (X, Y, Z=0).
    """
    cols, rows = chessboard_size
    objp = np.zeros((cols * rows, 3), np.float32)

    # cria grade de coordenadas (X, Y) e escala pelo tamanho do quadrado (mm)
    objp[:, :2] = np.mgrid[0:cols, 0:rows].T.reshape(-1, 2) * square_size_mm

    return objp


def calibrate_camera(imgpoints, chessboard_size, square_size_mm, image_size):
    """
    Executa a calibração da câmera relacionando os pontos 2D
    nas imagens com as coordenadas reais.

    :param imgpoints: Lista com os arrays de pontos 2D (subpixel) detectados.
    :param chessboard_size: Tupla (colunas, linhas) de cantos internos.
    :param square_size_mm: Tamanho do quadrado em mm.
    :param image_size: Tupla (largura, altura) da imagem em pixels.

    :return camera_matrix: Matriz intrínseca 3x3 da câmera (K).
    :return dist_coeffs: Vetor com os coeficientes de distorção da lente.
    :return rvecs: Lista de vetores de rotação para cada imagem válida.
    :return tvecs: Lista de vetores de translação para cada imagem válida.
    :return reprojection_error: Erro médio quadrático de reprojeção (RMS) em pixels.
    """
    print(f"[INFO] Gerando pontos 3D para {len(imgpoints)} imagens...")

    objp = generate_object_points(chessboard_size, square_size_mm)
    objpoints = [objp] * len(imgpoints)

    print("[INFO] Calculando matrizes de calibração...")

    # algoritmo de otimização
    camera_matrix, dist_coeffs = cv2.calibrateCamera(
        objpoints, imgpoints, image_size, None, None
    )

    print("[INFO] Calibração concluída com sucesso!")
    print(f"[INFO] Matriz Intrínseca (K):\n{camera_matrix}")
    print(f"[INFO] Coeficientes de Distorção:\n{dist_coeffs.ravel()}")
