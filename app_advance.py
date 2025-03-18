import carla
import cv2
import numpy as np
import time
import random
import sys


from camera_calibration import undistort,load_or_calibrate_camera
from image_processing import apply_thresholds, perspective_transform
from lane_detection import fit_polynomial, left_line, right_line
from visualization import draw_lane_overlay, add_metrics_to_image, get_lane_departure_warning, calculate_steering_correction,create_debug_view

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


# Configurações

    

LARGURA = 800
ALTURA = 600
LARGURA_FAIXA = 400
ALTURA_FAIXA = 200
FPS = 30

def main():
 
    client = carla.Client('localhost', 2000)
    client.set_timeout(10.0)
    
 
    world = client.get_world()
    print(f"A usar mapa: {world.get_map().name}")

    available_maps = client.get_available_maps()
    print("Mapas disponíveis:")
    for map_name in available_maps:
        print(f"  - {map_name}")
    

    #     Town01_Opt
    #   - /Game/Carla/Maps/Town04_Opt
    #   - /Game/Carla/Maps/Town03_Opt
    #   - /Game/Carla/Maps/Town10HD_Opt
    #   - /Game/Carla/Maps/Town05_Opt
    #   - /Game/Carla/Maps/Town02_Opt


    try:
        world = client.load_world('Town04_Opt')
        
        # if '/Game/Carla/Maps/Town04_Opt' in available_maps:
        #     world = client.load_world('Town04_Opt')
        #     print("Carregado mapa leve: Town04_Opt")
        # # Town05 que é mais aberto e geralmente mais leve
        # elif '/Game/Carla/Maps/Town05' in available_maps:
        #     world = client.load_world('Town05')
        #     print("Carregado mapa leve: Town05")
        # else:
        #     world = client.get_world()
        #     print(f"Usando mapa atual: {world.get_map().name}")
    except Exception as e:
        print(f"Erro ao carregar mapa: {e}")
        world = client.get_world()
        print(f"Usando mapa atual: {world.get_map().name}")
    

        world.unload_map_layer(carla.MapLayer.All)
        world.load_map_layer(carla.MapLayer.Ground)
        world.load_map_layer(carla.MapLayer.Buildings)
        world.load_map_layer(carla.MapLayer.Props)
        world.load_map_layer(carla.MapLayer.StreetLights)
    
    # Configurações do modo síncrono
    settings = world.get_settings()
    modo_sincrono = False
    
    if not settings.synchronous_mode:
        modo_sincrono = True
        settings.synchronous_mode = True
        settings.fixed_delta_seconds = 0.05
        world.apply_settings(settings)
        print("Modo síncrono ativado")
    
 
    lista_atores = []
    
    try:
 
        blueprint_library = world.get_blueprint_library()
        veiculo_bp = blueprint_library.find('vehicle.tesla.model3')
        
        pontos_spawn = world.get_map().get_spawn_points()
        if not pontos_spawn:
            print("Não há pontos de spawn disponíveis!")
            return
        
        ponto_spawn = random.choice(pontos_spawn)
        #caso queremos o random
        print("Ponto:",ponto_spawn)
        #vou escolher este , se mudar o mapa tem que ser ouro
        #ponto_spawn=carla.Transform(carla.Location(x=408.696899, y=-32.305439, z=0.281942), carla.Rotation(pitch=0.000000, yaw=-89.560913, roll=0.000000))
        ponto_spawn=carla.Transform(carla.Location(x=410.947449, y=-14.670991, z=0.281942), carla.Rotation(pitch=0.000000, yaw=-79.304489, roll=0.000000))# wown4pt

        
        
        print("A criar veículo...")
        veiculo = world.spawn_actor(veiculo_bp, ponto_spawn)
        lista_atores.append(veiculo)
        print(f"Veículo criado: {veiculo.type_id}")
        
        # Configurar câmara principal
        camera_bp = blueprint_library.find('sensor.camera.rgb')
        camera_bp.set_attribute('image_size_x', str(LARGURA))
        camera_bp.set_attribute('image_size_y', str(ALTURA))
        camera_bp.set_attribute('fov', '90')
        camera_bp.set_attribute('sensor_tick', '0.05')  # 20 FPS
        
        transformacao_camera = carla.Transform(
            carla.Location(x=2.0, z=1.4),
            carla.Rotation(pitch=-15)
        )
        
        camera = world.spawn_actor(camera_bp, transformacao_camera, attach_to=veiculo)
        lista_atores.append(camera)
        print("Câmara criada")
        
        #  imagems
        imagem_atual = None

        

        cv2.namedWindow('CARLA', cv2.WINDOW_NORMAL)
     

        try:
            mtx, dist = load_or_calibrate_camera()
        except Exception as e:
            print(f"Erro ao calibrar a câmera: {e}")
            exit(1)
        
        # Definir pontos de origem e destino para transformação de perspectiva
        # Estes valores são estimativas iniciais e podem precisar de ajustes para cada pista
        src = np.float32([
        [280, 720],   # Inferior esquerdo
        [550, 480],   # Superior esquerdo
        [730, 480],   # Superior direito
        [1000, 720]   # Inferior direito
        ])
        
        dst = np.float32([
            [320, 720],   # Inferior esquerdo
            [320, 0],     # Superior esquerdo
            [960, 0],     # Superior direito
            [960, 720]    # Inferior direito
        ])
        
 
        # Função para processar a imagem da câmara
        def processar_imagem(imagem):
            nonlocal imagem_atual
            # Converter para formato OpenCV
            array = np.frombuffer(imagem.raw_data, dtype=np.dtype("uint8"))
            array = np.reshape(array, (imagem.height, imagem.width, 4))
            imagem_atual = array[:, :, :3].copy()
            
           
        
        # Registar a função de callback
        camera.listen(processar_imagem)
        
        # Controlos do carro
        controlo_atual = carla.VehicleControl()
        autopilot_enabled = False
        traffic_manager = client.get_trafficmanager()
        traffic_manager.set_synchronous_mode(modo_sincrono)
        traffic_manager.set_global_distance_to_leading_vehicle(2.5)
        # Isto faz com que o veículo siga as regras de trânsito
        # Valores mais baixos são mais agressivos
        traffic_manager.global_percentage_speed_difference(10.0)
        a_correr = True
        print("Controlos: W/S - Acelerar/Travar | A/D - Virar | ESPAÇO - Travão de mão | ESC - Sair")
        
        while a_correr:
            if modo_sincrono:
                world.tick()
            else:
                time.sleep(0.05)  #  não sobrecarregar o CPU
            
            # Mostrar as imagens
            if imagem_atual is not None:
          
                
                # Converter imagem original para BGR 
                img_original =imagem_atual.copy()# cv2.cvtColor(imagem_atual, cv2.COLOR_RGB2BGR)


                result_dict = process_image(imagem_atual, mtx, dist, src, dst)
                result = result_dict['result']
    

            
                # Obter todas as visualizações intermediárias
                debug_view = create_debug_view(
                            img_original,
                            result_dict['undistorted'],
                            result_dict['binary'],
                            result_dict['warped'],
                            result_dict['lane_img'],
                            result_dict['detection_img'],
                            result
                        )
            
                
                # Adicionar informações à imagem original
                velocidade = veiculo.get_velocity()
                rapidez = 3.6 * np.sqrt(velocidade.x**2 + velocidade.y**2 + velocidade.z**2)  # km/h
                info = f"Vel: {rapidez:.1f} km/h | Steer: {controlo_atual.steer:.2f}"
                cv2.putText(img_original, info, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                if not autopilot_enabled:
                    cv2.putText(img_original, "ESC: Sair | W/S: Acel/Trav | A/D: Virar", (10, ALTURA - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                else:
                    cv2.putText(img_original, "ESC: Sair | Autopilot", (10, ALTURA - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                
                
 
          
                
              

                
             
                cv2.imshow('Debug', img_original)

                   # Mostrar o resultado final
                cv2.imshow('CARLA', debug_view)
                
                # Mostrar visualizações intermediárias em janelas separadas
                # cv2.imshow('Imagem Original', frame)
                # cv2.imshow('Imagem Corrigida', result_dict['undistorted'])
                #cv2.imshow('Threshold', result_dict['binary'] * 255)
                #cv2.imshow('Topo (Warped)', result_dict['warped'] * 255)
                
                # if result_dict['lane_img'] is not None:
                #     cv2.imshow('Detect', result_dict['lane_img'])
                
                # if result_dict['detection_img'] is not None:
                #     cv2.imshow('Janelas Deslizantes', result_dict['detection_img'])
                
        
                key = cv2.waitKey(1) & 0xFF
                
       
                if key == 27:  # ESC
                    a_correr = False
                elif key == ord('p'):  # Tecla 'p' para ativar/desativar piloto automático
                    # Alterna o estado do piloto automático
                    autopilot_enabled = not autopilot_enabled
                    veiculo.set_autopilot(autopilot_enabled)
                    print(f"Piloto automático {'ativado' if autopilot_enabled else 'desativado'}")
                elif key == ord('w'):
                    controlo_atual.throttle = 0.7
                    controlo_atual.brake = 0.0
                elif key == ord('s'):
                    controlo_atual.throttle = 0.0
                    controlo_atual.brake = 0.7
                elif key == ord('a'):
                    controlo_atual.steer = -0.5
                elif key == ord('d'):
                    controlo_atual.steer = 0.5
                elif key == ord(' '):  # spce
                    controlo_atual.hand_brake = not controlo_atual.hand_brake
                else:
 
                    if key not in [ord('w'), ord('s')]:
                        controlo_atual.throttle = 0.0
                        controlo_atual.brake = 0.0
                    if key not in [ord('a'), ord('d')]:
                        controlo_atual.steer = 0.0
                
                if not autopilot_enabled:
                    veiculo.apply_control(controlo_atual)
    
    finally:
        if 'modo_sincrono' in locals() and modo_sincrono:
            settings = world.get_settings()
            settings.synchronous_mode = False
            settings.fixed_delta_seconds = None
            world.apply_settings(settings)
        
        if 'lista_atores' in locals():
            print("A limpar atores...")
            for ator in lista_atores:
                if ator is not None and ator.is_alive:
                    ator.destroy()
        

        cv2.destroyAllWindows()
        print("Simulação terminada")

if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print("Exit")
        cv2.destroyAllWindows()
    except Exception as e:
        print(f"Erro: {e}")
        cv2.destroyAllWindows()