import numpy as np
import cv2

def apply_thresholds(img):
    """
    Aplica diversos limiares para detectar bordas e faixas
    Retorna uma imagem binária onde as faixas são destacadas
    """
    # Converter para escala de cinza
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Aplicar blur para reduzir ruído
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # Aplicar detector de bordas Canny
    edges = cv2.Canny(blur, 50, 150)
    
    # Aplicar limiar adaptativo
    thresh = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
    
    # Converter para espaço de cor HLS
    hls = cv2.cvtColor(img, cv2.COLOR_BGR2HLS)
    
    # Separar os canais
    h_channel = hls[:,:,0]
    l_channel = hls[:,:,1]
    s_channel = hls[:,:,2]
    
    # Aplicar limiar ao canal S para detectar linhas amarelas e brancas
    s_thresh_min = 100
    s_thresh_max = 255
    s_binary = np.zeros_like(s_channel)
    s_binary[(s_channel >= s_thresh_min) & (s_channel <= s_thresh_max)] = 1
    
    # Aplicar limiar ao canal L para detectar linhas brancas
    l_thresh_min = 180
    l_thresh_max = 255
    l_binary = np.zeros_like(l_channel)
    l_binary[(l_channel >= l_thresh_min) & (l_channel <= l_thresh_max)] = 1
    
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    abs_sobelx = np.absolute(sobelx)
    scaled_sobel = np.uint8(255*abs_sobelx/np.max(abs_sobelx))

    # Aplicar limiar ao gradiente
    thresh_min = 20
    thresh_max = 100
    sobel_binary = np.zeros_like(scaled_sobel)
    sobel_binary[(scaled_sobel >= thresh_min) & (scaled_sobel <= thresh_max)] = 1

    # Combinar com os outros canais
    combined_binary = np.zeros_like(s_binary)
    combined_binary[(l_binary == 1) | (sobel_binary == 1)] = 1

    # Combinar os resultados
    #combined_binary[(s_binary == 1) | (l_binary == 1)] = 1
    #combined_binary[l_binary == 1] = 1
    
    # Aplicar máscara para focar apenas na região de interesse
    mask = np.zeros_like(combined_binary)
    height, width = combined_binary.shape
    region = np.array([[
    (width*0.2, height),
    (width*0.45, height*0.65),  # Ajustar para a estrada portuguesa
    (width*0.55, height*0.65),
    (width*0.8, height)
        ]], dtype=np.int32)
    cv2.fillPoly(mask, region, 1)
    masked_binary = cv2.bitwise_and(combined_binary, mask)
    
    return masked_binary

def perspective_transform(img, src=None, dst=None):
    """
    Aplica transformação de perspectiva para obter visão de cima da estrada
    """
    img_size = (img.shape[1], img.shape[0])
    
    if src is None:
        # Pontos de origem padrão
        src = np.float32([
            [190, 720],   # Inferior esquerdo
            [590, 450],   # Superior esquerdo
            [690, 450],   # Superior direito
            [1130, 720]   # Inferior direito
        ])
    
    if dst is None:
        # Pontos de destino padrão
        dst = np.float32([
            [320, 720],   # Inferior esquerdo
            [320, 0],     # Superior esquerdo
            [960, 0],     # Superior direito
            [960, 720]    # Inferior direito
        ])
    
    # Calcular matriz de transformação e sua inversa
    M = cv2.getPerspectiveTransform(src, dst)
    Minv = cv2.getPerspectiveTransform(dst, src)
    
    # Aplicar transformação de perspectiva
    warped = cv2.warpPerspective(img, M, img_size, flags=cv2.INTER_LINEAR)
    
    return warped, M, Minv
