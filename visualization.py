import cv2
import numpy as np
from lane_detection import left_line, right_line

def draw_lane_overlay(undist, binary_warped, left_fitx, right_fitx, ploty, Minv):
    """
    Desenhar a faixa detectada de volta na imagem original
    """
    # Criar uma imagem em branco para desenhar as faixas
    warp_zero = np.zeros_like(binary_warped).astype(np.uint8)
    color_warp = np.dstack((warp_zero, warp_zero, warp_zero))
    
    if left_fitx is not None and right_fitx is not None:
        # Reformatar arrays para desenhar polígono
        pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
        pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
        pts = np.hstack((pts_left, pts_right))
        
        # Desenhar o polígono da faixa
        cv2.fillPoly(color_warp, np.int_([pts]), (0, 255, 0))
        
        # Desenhar as linhas da faixa
        for i in range(1, len(ploty), 10):
            if i < len(ploty):
                # Linha esquerda em vermelho
                if i < len(left_fitx):
                    pt1 = (int(left_fitx[i-1]), int(ploty[i-1]))
                    pt2 = (int(left_fitx[i]), int(ploty[i]))
                    cv2.line(color_warp, pt1, pt2, (255, 0, 0), 10)
                
                # Linha direita em azul
                if i < len(right_fitx):
                    pt1 = (int(right_fitx[i-1]), int(ploty[i-1]))
                    pt2 = (int(right_fitx[i]), int(ploty[i]))
                    cv2.line(color_warp, pt1, pt2, (0, 0, 255), 10)
    
    # Aplicar transformação inversa
    newwarp = cv2.warpPerspective(color_warp, Minv, (undist.shape[1], undist.shape[0]))
    
    # Combinar resultado com a imagem original
    result = cv2.addWeighted(undist, 1, newwarp, 0.3, 0)
    
    return result

def add_metrics_to_image(img, avg_curvature, vehicle_offset):
    """
    Adicionar métricas (raio de curvatura e posição do veículo) à imagem
    """
    font = cv2.FONT_HERSHEY_SIMPLEX
    
    # Adicionar texto com raio de curvatura
    if avg_curvature is not None:
        text = f'Radius of Curvature: {avg_curvature:.1f}m'
        cv2.putText(img, text, (50, 50), font, 1, (255, 255, 255), 2)
    
    # Adicionar texto com posição do veículo
    if vehicle_offset is not None:
        direction = 'right' if vehicle_offset < 0 else 'left'
        text = f'Vehicle position: {abs(vehicle_offset):.2f}m {direction} of center'
        cv2.putText(img, text, (50, 100), font, 1, (255, 255, 255), 2)
    
    return img

def get_lane_departure_warning(offset):
    """
    Determinar alertas baseados no offset do veículo
    """
    if offset is None:
        return None
    
    abs_offset = abs(offset)
    if abs_offset < 0.3:
        return None  # Dentro da zona segura
    elif abs_offset < 0.6:
        return "Atenção: Aproximando-se da borda da faixa"
    else:
        return "ALERTA: Saindo da faixa!"

def calculate_steering_correction(offset, speed=60):
    """
    Calcular correção de direção baseada no offset
    """
    if offset is None:
        return 0
    
    # Quanto maior a velocidade, mais suave deve ser a correção
    correction_factor = 1.0 / max(50, speed)
    # Inverte o sinal (offset positivo requer virar à direita)
    steering_angle = -offset * correction_factor * 5.0
    
    return steering_angle

def create_debug_view(original, undistorted, binary, warped, lane_img, detection_img, result):
    """
    Criar uma visualização de debug mostrando várias etapas do processamento
    """
    # Converter imagens binárias para 3 canais
    binary_3ch = np.dstack((binary*255, binary*255, binary*255)).astype(np.uint8)
    warped_3ch = np.dstack((warped*255, warped*255, warped*255)).astype(np.uint8)
    
    # Redimensionar todas as imagens para o mesmo tamanho
    h, w = original.shape[:2]
    scaled_h, scaled_w = h // 2, w // 2
    
    # Redimensionar cada imagem
    original_small = cv2.resize(original, (scaled_w, scaled_h))
    undistorted_small = cv2.resize(undistorted, (scaled_w, scaled_h))
    binary_small = cv2.resize(binary_3ch, (scaled_w, scaled_h))
    warped_small = cv2.resize(warped_3ch, (scaled_w, scaled_h))
    
    # Verificar se as imagens de detecção existem
    if lane_img is not None and detection_img is not None:
        lane_small = cv2.resize(lane_img, (scaled_w, scaled_h))
        detection_small = cv2.resize(detection_img, (scaled_w, scaled_h))
    else:
        # Criar imagens vazias se não existirem
        lane_small = np.zeros((scaled_h, scaled_w, 3), dtype=np.uint8)
        detection_small = np.zeros((scaled_h, scaled_w, 3), dtype=np.uint8)
    
    result_small = cv2.resize(result, (scaled_w, scaled_h))
    
    # Criar layout de 4x2 imagens
    top_row = np.hstack((original_small, undistorted_small, binary_small, warped_small))
    bottom_row = np.hstack((lane_small, detection_small, result_small, np.zeros_like(result_small)))
    
    debug_view = np.vstack((top_row, bottom_row))
    
    # Adicionar texto para cada imagem
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.5
    font_color = (255, 255, 255)
    font_thickness = 1
    
    labels = [
        "Original", "Undistorted", "Binary Threshold", "Warped",
        "Lane Detection", "Sliding Windows", "Final Result", ""
    ]
    
    for i, label in enumerate(labels):
        x = (i % 4) * scaled_w + 10
        y = (i // 4) * scaled_h + 30
        cv2.putText(debug_view, label, (x, y), font, font_scale, font_color, font_thickness)
    
    return debug_view
