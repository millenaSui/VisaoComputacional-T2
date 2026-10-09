# TA02 - Visão Computacional (Calibração de Câmera)

A aplicação foi projetada para receber um conjunto de imagens de um padrão geométrico plano (tabuleiro de xadrez digital ou impresso), convertê-las para escala de cinza (*grayscale*), identificar a posição dos seus cantos internos e, através da metodologia de calibração utilizada no OpenCV, estimar os parâmetros geométricos e os coeficientes de distorção óptica da câmera, permitindo a correção de distorções ópticas em imagens.

## Fluxo

O fluxo de trabalho da aplicação é dividido nas seguintes etapas:

- **Leitura e Pré-Processamento (`format_images`):** As fotografias do tabuleiro de xadrez presentes no diretório `./images/dataset/` são carregadas, convertidas para *grayscale* e organizadas junto às suas dimensões de resolução original.

## Execução

Devido às dependências de sistema exigidas pelo `OpenCV` (como bibliotecas gráficas nativas e suporte matemático), a aplicação foi conteinerizada via *Docker*, afim de garantir um ambiente de execução isolado, eliminando a necessidade de configurar manualmente o ambiente *Python* ou instalar pacotes de sistema na máquina *host*.

Para executar o projeto (no diretório raiz e com a ferramenta *Docker* devidamente instalada):

```bash
docker build -t calibracao-camera .
docker run --rm -v $(pwd)/images/dataset:/app/images/dataset -v $(pwd):/app/out calibracao-camera
```