# TA02 - Visão Computacional (Calibração de Câmera)

A aplicação foi projetada para receber um conjunto de imagens de um padrão geométrico plano (tabuleiro de xadrez digital ou impresso), convertê-las para escala de cinza (*grayscale*), identificar a posição dos seus cantos internos e, através da metodologia de calibração utilizada no OpenCV, estimar os parâmetros geométricos e os coeficientes de distorção óptica da câmera, permitindo a correção de distorções em imagens.

## Fluxo de Trabalho

O fluxo de execução da aplicação é automatizado e dividido nas seguintes etapas:

1. **Leitura e Pré-Processamento:** As fotografias do tabuleiro de xadrez, que devem estar presentes no diretório `./images/dataset/`, são carregadas, convertidas para *grayscale* e organizadas junto às suas dimensões de resolução original.
2. **Detecção de Características:** A aplicação localiza os cantos internos do tabuleiro em cada imagem (`cv2.findChessboardCorners`) e aplica um refinamento subpixel para aumentar a precisão da detecção (`cv2.cornerSubPix`).
3. **Calibração da Câmera:** Utilizando os pontos 2D encontrados e as coordenadas 3D reais (calculadas com base no tamanho do quadrado de 25mm), o algoritmo calcula a matriz intrínseca da câmera e seus coeficientes de distorção (`cv2.calibrateCamera`).
4. **Remoção de Distorção (Undistort):** Com os parâmetros intrínsecos estimados, a distorção radial e tangencial é removida das imagens originais. Um comparativo lado a lado (Antes x Depois) é gerado para facilitar a visualização.
5. **Validação (Experimento de Reprojeção):** Os pontos 3D ideais são reprojetados no plano 2D da imagem original para calcular o erro médio de reprojeção. Imagens anotadas (com pontos detectados vs. reprojetados) são salvas para atestar a qualidade da calibração.

## Execução

Devido às dependências de sistema exigidas pelo `OpenCV` (como bibliotecas gráficas nativas e suporte matemático), a aplicação foi conteinerizada via **Docker**, a fim de garantir um ambiente de execução isolado, eliminando a necessidade de configurar manualmente o ambiente *Python* ou instalar pacotes de sistema na máquina *host*.

### Pré-requisitos
Certifique-se de ter o [Docker](https://docs.docker.com/get-docker/) instalado e as imagens de calibração na pasta `./images/dataset/`.

### Usando Docker Compose (Recomendado)
A maneira mais fácil de executar o projeto é utilizando o Docker Compose, que já mapeia automaticamente os volumes corretos:

```bash
docker-compose up --build