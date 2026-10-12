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
    ret, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(
        objpoints, imgpoints, image_size, None, None
    )

    print("[INFO] Calibração concluída com sucesso!")
    print(f"[INFO] Matriz Intrínseca (K):\n{camera_matrix}")
    print(f"[INFO] Coeficientes de Distorção:\n{dist_coeffs.ravel()}")

    return camera_matrix, dist_coeffs, rvecs, tvecs, ret



def remove_distortion(formatted_images, camera_matrix, dist_coeffs, undistorted_dir):
    """
    Aplica a correção de distorção radial e tangencial nas imagens
    e salva os resultados individuais e comparativos lado a lado.

    :param formatted_images: Lista de tuplas (caminho_da_imagem, imagem_grayscale, imagem_original_bgr).
    :param camera_matrix: Matriz intrínseca 3x3 da câmera (K).
    :param dist_coeffs: Vetor de coeficientes de distorção da lente.
    :param output_dir: Diretório onde as imagens processadas serão salvas.

    :return undistorted_images: Lista de tuplas (caminho_original, imagem_corrigida_bgr).
    """
    if undistorted_dir and not os.path.exists(undistorted_dir):
        os.makedirs(undistorted_dir)
        print(f"[INFO] Diretório '{undistorted_dir}' criado para armazenar as imagens corrigidas.")

    undistorted_images = []
    total_images = len(formatted_images)

    print(f"[INFO] Iniciando remoção de distorção em {total_images} imagens...")

    h, w = formatted_images[0][1].shape[:2] # dimensoes da imagem
    image_size = (w, h)
    new_camera_matrix, _ = cv2.getOptimalNewCameraMatrix(
        camera_matrix, dist_coeffs, image_size, alpha=1, newImgSize=image_size
    )

    for fname, _, img_bgr in formatted_images:
        # remoção da distorção
        undistorted_bgr = cv2.undistort(
            img_bgr, camera_matrix, dist_coeffs, None, new_camera_matrix
        )

        undistorted_images.append((fname, undistorted_bgr))

        if undistorted_dir:
            base_name = os.path.basename(fname)
            name_no_ext, ext = os.path.splitext(base_name)

            # salva a comparação de imagens lado a lado
            comparison = np.hstack((img_bgr, undistorted_bgr))
            cv2.putText(comparison, "Original (Com Distorcao)", (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2, cv2.LINE_AA)
            cv2.putText(comparison, "Corrigida (Sem Distorcao)", (w + 20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2, cv2.LINE_AA)

            out_path_comp = os.path.join(undistorted_dir, f"{name_no_ext}_comparison{ext}")
            cv2.imwrite(out_path_comp, comparison)

    print(f"[INFO] Remoção de distorção concluída: {len(undistorted_images)} imagens salvas em '{undistorted_dir}'.\n")
    return undistorted_images


def run_projection_experiment(valid_images, imgpoints, camera_matrix, dist_coeffs, 
                              rvecs, tvecs, chessboard_size, square_size_mm, 
                              output_dir='images/experiment'):
    """
    Executa o experimento de projeção perspectiva 3D -> 2D para validar a calibração.

    :param valid_images: Lista de tuplas (caminho_da_imagem, imagem_grayscale, imagem_original_bgr).
    :param imgpoints: Lista com os arrays de pontos 2D (subpixel) detectados.
    :param camera_matrix: Matriz intrínseca 3x3 da câmera (K).
    :param dist_coeffs: Vetor de coeficientes de distorção.
    :param rvecs: Vetores de rotação da calibração.
    :param tvecs: Vetores de translação da calibração.
    :param chessboard_size: Tupla (colunas, linhas) de cantos internos.
    :param square_size_mm: Tamanho do quadrado do tabuleiro em mm.
    :param output_dir: Diretório para salvar as imagens do experimento.

    :return experiment_results: Lista de dicionários com estatísticas de erro por imagem.
    """
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"[INFO] Diretório '{output_dir}' criado para o experimento de projeção.")

    # define os pontos 3D do tabuleiro (plano Z=0)
    cols, rows = chessboard_size
    objp = np.zeros((cols * rows, 3), np.float32)
    objp[:, :2] = np.mgrid[0:cols, 0:rows].T.reshape(-1, 2) * square_size_mm

    experiment_results = []
    print(f"[INFO] Iniciando experimento 3D -> 2D em {len(valid_images)} imagens...")

    for i, (fname, _, img_bgr) in enumerate(valid_images):
        # projeção 2D pra 3D
        projected_imgpoints, _ = cv2.projectPoints(
            objp, rvecs[i], tvecs[i], camera_matrix, dist_coeffs
        )

        detected_pts = imgpoints[i].reshape(-1, 2)
        projected_pts = projected_imgpoints.reshape(-1, 2)

        # cálculo do erro em pixels
        errors = np.linalg.norm(detected_pts - projected_pts, axis=1)
        mean_error = np.mean(errors)
        max_error = np.max(errors)

        experiment_results.append({
            'image_path': fname,
            'mean_error_px': mean_error,
            'max_error_px': max_error,
            'projected_pts': projected_pts,
            'detected_pts': detected_pts
        })

        # gera imagem anotada para o relatório
        if output_dir:
            img_annotated = img_bgr.copy()
            for det, proj in zip(detected_pts, projected_pts):
                cv2.circle(img_annotated, (int(det[0]), int(det[1])), 4, (0, 0, 255), -1) # ponto detectado círculo
                cv2.drawMarker(img_annotated, (int(proj[0]), int(proj[1])), (0, 255, 0), # ponto projetado cruz
                               markerType=cv2.MARKER_CROSS, markerSize=8, thickness=1)

            base_name = os.path.basename(fname)
            out_path = os.path.join(output_dir, f"proj_{base_name}")
            cv2.imwrite(out_path, img_annotated)

    avg_all_images = np.mean([r['mean_error_px'] for r in experiment_results])
    print(f"[INFO] Experimento concluído! Erro médio de projeção global: {avg_all_images:.4f} pixels.")
    print(f"[INFO] Imagens com marcas salvas em '{output_dir}'.\n")

    return experiment_results