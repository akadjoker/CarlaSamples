import cv2
import numpy as np
from camera_calibration import undistort
from image_processing import apply_thresholds, perspective_transform
from lane_detection import fit_polynomial, left_line, right_line
from visualization import draw_lane_overlay, add_metrics_to_image, get_lane_departure_warning, calculate_steering_correction

def process_image(img, mtx, dist, src=None, dst=None):
    """
    Função principal para processar cada frame do vídeo
    """
    # Corrigir distorção
    undistorted = undistort(img, mtx, dist)
    
    # Aplicar limiares para detectar bordas e faixas
    binary = apply_thresholds(undistorted)
    
    # Aplicar transformação de perspectiva
    warped, M, Minv = perspective_transform(binary, src, dst)
    
    # Detectar faixas e ajustar polinômios
    left_fitx, right_fitx, lane_img, detection_img, ploty = fit_polynomial(warped)
    
    # Se a detecção falhou, usar valores anteriores ou retornar imagem original
    if left_fitx is None or right_fitx is None:
        result = undistorted
        return {
            'result': result,
            'undistorted': undistorted,
            'binary': binary,
            'warped': warped,
            'lane_img': None,
            'detection_img': None,
            'left_fitx': None,
            'right_fitx': None,
            'metrics': {
                'curvature': None,
                'offset': None,
                'warning': None,
                'steering': None
            }
        }
    else:
        # Desenhar faixa na imagem original
        result = draw_lane_overlay(undistorted, binary, left_fitx, right_fitx, ploty, Minv)
        
        # Calcular médias das métricas
        avg_curvature = (left_line.radius_of_curvature + right_line.radius_of_curvature) / 2
        vehicle_offset = (left_line.line_base_pos + right_line.line_base_pos) / 2
        
        # Adicionar métricas à imagem
        result = add_metrics_to_image(result, avg_curvature, vehicle_offset)
        
        # Verificar alerta de saída de faixa
        warning = get_lane_departure_warning(vehicle_offset)
        if warning:
            cv2.putText(result, warning, (50, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        
        # Calcular correção de direção
        steering = calculate_steering_correction(vehicle_offset)
        cv2.putText(result, f'Steering correction: {steering:.2f} degrees', (50, 200), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
        # Retornar resultados em um dicionário
        return {
            'result': result,
            'undistorted': undistorted,
            'binary': binary,
            'warped': warped,
            'lane_img': lane_img,
            'detection_img': detection_img,
            'left_fitx': left_fitx,
            'right_fitx': right_fitx,
            'metrics': {
                'curvature': avg_curvature,
                'offset': vehicle_offset,
                'warning': warning,
                'steering': steering
            }
        }

def process_video(input_path, output_path, mtx, dist, src=None, dst=None, debug_output_path=None):
    """
    Processar um vídeo completo usando OpenCV
    """
    # Abrir o vídeo de entrada
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        print(f"Erro ao abrir o vídeo: {input_path}")
        return
    
    # Obter informações do vídeo
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Configurar o vídeo de saída
    out = cv2.VideoWriter(
        output_path,
        cv2.VideoWriter_fourcc(*'XVID'),
        fps,
        (width, height)
    )
    
    # Configurar o vídeo de debug (se necessário)
    debug_out = None
    if debug_output_path:
        debug_out = cv2.VideoWriter(
            debug_output_path,
            cv2.VideoWriter_fourcc(*'XVID'),
            fps,
            (width*2, height*2)  # Layout de debug é 2x maior
        )
    
    # Inicializar contador de frames
    frame_num = 0
    
    # Processar cada frame do vídeo
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        # Processar o frame
        frame_num += 1
        print(f"Processando frame {frame_num}/{frame_count}")
        
        # Processar imagem
        result_dict = process_image(frame, mtx, dist, src, dst)
        result = result_dict['result']
        
        # Escrever no vídeo de saída
        out.write(result)
        
        # Se o debug estiver habilitado, criar visualização de debug
        if debug_out is not None:
            from visualization import create_debug_view
            
            # Obter todas as visualizações intermediárias
            debug_view = create_debug_view(
                frame,
                result_dict['undistorted'],
                result_dict['binary'],
                result_dict['warped'],
                result_dict['lane_img'],
                result_dict['detection_img'],
                result
            )
            
            # Escrever no vídeo de debug
            debug_out.write(debug_view)
    
    # Liberar recursos
    cap.release()
    out.release()
    if debug_out is not None:
        debug_out.release()
    
    print(f"Processamento concluído. Vídeo salvo como {output_path}")
    if debug_output_path:
        print(f"Visualização de debug salva como {debug_output_path}")
