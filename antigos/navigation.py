import carla
import cv2
import numpy as np
import time
import random
import sys
import os
import datetime
import math
import argparse

#   caminho do PythonAPI do CARLA
 
sys.path.append('/home/djoker/code/CARLA_0.9.15/PythonAPI/carla')
from agents.navigation.global_route_planner import GlobalRoutePlanner


# Variáveis para controle de coleta de dados
collecting_dataset = False
dataset_dir = None
dataset_images_dir = None
dataset_file = None
frame_count = 0
current_image = None

# Configurações de exibição
WIDTH = 800
HEIGHT = 600

def create_dataset_session():
    """Cria uma nova sessão para armazenar dados (imagens e valores de direção)"""
    global dataset_dir, dataset_images_dir, frame_count, dataset_file
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    dataset_dir = f"dataset/session_{timestamp}"
    dataset_images_dir = f"{dataset_dir}/images"
    os.makedirs(dataset_images_dir, exist_ok=True)
    
    dataset_file = open(f"{dataset_dir}/steering_data.csv", "w")
    dataset_file.write("image_path,steering\n")
    
    frame_count = 0
    print(f"Nova sessão de dataset criada: {dataset_dir}")
    return dataset_dir

def toggle_dataset_collection():
    """Ativa/desativa a coleta de dados"""
    global collecting_dataset, dataset_file
    if not collecting_dataset:
        os.makedirs("dataset", exist_ok=True)
        create_dataset_session()
        collecting_dataset = True
        print("Iniciando coleta de dados")
    else:
        if dataset_file:
            dataset_file.close()
            dataset_file = None
        collecting_dataset = False
        print(f"Coleta de dados finalizada. Total de frames: {frame_count}")

def save_frame_to_dataset(frame, steering):
    """Salva um frame único com seu valor de direção para o dataset"""
    global dataset_file, frame_count
    if not dataset_file:
        print("Coleta de dados desligada.")
        return False
        
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    image_filename = f"frame_{timestamp}.jpg"
    image_path = f"{dataset_images_dir}/{image_filename}"
    cv2.imwrite(image_path, frame)
    
    dataset_file.write(f"images/{image_filename},{steering:.6f}\n")
    dataset_file.flush()
    
    frame_count += 1
    if frame_count % 10 == 0:
        print(f"Frames capturados: {frame_count}", end="\r")
    
    return True

def get_route(world, start_point, end_point, sampling_resolution=1.0):
    """
    Usa o GlobalRoutePlanner para obter uma rota entre dois pontos
    """
    grp = GlobalRoutePlanner(world.get_map(), sampling_resolution)
    route = grp.trace_route(start_point.location, end_point.location)
    print(f"Rota gerada com {len(route)} waypoints")
    return route

def get_longest_route(world, start_point, spawn_points, sampling_resolution=1.0):
    """
    Encontra a rota mais longa possível a partir do ponto inicial
    """
    grp = GlobalRoutePlanner(world.get_map(), sampling_resolution)
    longest_route = None
    max_length = 0
    
    for end_point in spawn_points:
        if end_point.location.distance(start_point.location) < 10:
            continue  # Ignorar pontos muito próximos
        
        route = grp.trace_route(start_point.location, end_point.location)
        if len(route) > max_length:
            max_length = len(route)
            longest_route = route
            longest_end_point = end_point
    
    print(f"Rota mais longa encontrada com {max_length} waypoints")
    return longest_route, longest_end_point

def process_image(image):
    """Função de callback para processar imagens da câmera"""
    global current_image
    array = np.frombuffer(image.raw_data, dtype=np.dtype("uint8"))
    array = np.reshape(array, (image.height, image.width, 4))
    current_image = array[:, :, :3].copy()

def drive_route(world, vehicle, route, direction_forward=True, speed_factor=0.5):
    """
    Dirige o veículo seguindo a rota definida.
    A captura de dados ocorre manualmente quando o usuário pressiona Enter.
    
    Parâmetros:
    - world: objeto mundo do CARLA
    - vehicle: ator do veículo para controlar
    - route: Lista de waypoints para seguir
    - direction_forward: Se False, a rota é seguida em reverso
    - speed_factor: Controla velocidade do veículo (0.1-1.0)
    
    Retorna:
    - True se a rota foi completada, False se o usuário saiu
    """
    global current_image
    
    # Se a direção for invertida, inverte os waypoints
    waypoints = [wp[0] for wp in route]  # Extrair apenas os waypoints, não o tipo de conexão
    if not direction_forward:
        waypoints = waypoints[::-1]
    
    # Configurações iniciais
    current_waypoint_index = 0
    total_waypoints = len(waypoints)
    vehicle_control = carla.VehicleControl()
    
    # Loop principal
    running = True
    paused = False
    
    print("\n=== NAVEGAÇÃO INICIADA ===")
    print(f"Direção: {'PARA FRENTE' if direction_forward else 'REVERSA'}")
    print("ENTER: Capturar frame | P: Pausar | +/-: Velocidade | ESC: Sair")
    
    while running and current_waypoint_index < total_waypoints:
        # Atualiza o mundo no modo síncrono
        world.tick()
        
        # Obtém o próximo waypoint
        target_waypoint = waypoints[current_waypoint_index]
        
        if not paused:
            # Calcula posições do veículo e waypoint
            vehicle_transform = vehicle.get_transform()
            vehicle_location = vehicle_transform.location
            target_location = target_waypoint.transform.location
            
            # Calcula direção e distância
            direction = target_location - vehicle_location
            distance = math.sqrt(direction.x**2 + direction.y**2)
            
            # Se estiver perto o suficiente, avança para o próximo waypoint
            if distance < 2.0:  # 2 metros de tolerância
                current_waypoint_index += 1
                if current_waypoint_index % 10 == 0:
                    print(f"Waypoint {current_waypoint_index}/{total_waypoints}")
                if current_waypoint_index >= total_waypoints:
                    print("Rota completada!")
                    break
                continue
            
            # Calcula ângulo entre veículo e waypoint
            vehicle_forward = vehicle_transform.get_forward_vector()
            dot = vehicle_forward.x * direction.x + vehicle_forward.y * direction.y
            cross = vehicle_forward.x * direction.y - vehicle_forward.y * direction.x
            angle = math.atan2(cross, dot)
            
            # Converte ângulo para valor de direção [-1, 1]
            steering = max(-1.0, min(1.0, angle * 2.0))
            
            # Aplica controle do veículo
            vehicle_control.throttle = speed_factor
            vehicle_control.steer = steering
            vehicle_control.brake = 0.0
            vehicle.apply_control(vehicle_control)
        
        # Mostra imagens
        if current_image is not None:
            img_display = current_image.copy()
            
            # Adiciona informações à imagem
            velocity = vehicle.get_velocity()
            speed = 3.6 * np.sqrt(velocity.x**2 + velocity.y**2 + velocity.z**2)  # km/h
            current_steering = vehicle_control.steer
            
            # Informações básicas
            cv2.putText(img_display, f"Velocidade: {speed:.1f} km/h", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(img_display, f"Direção: {current_steering:.2f}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(img_display, f"Waypoint: {current_waypoint_index}/{total_waypoints}", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(img_display, f"Sentido: {'FRENTE' if direction_forward else 'REVERSO'}", (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            # Informações do dataset
            if collecting_dataset:
                cv2.putText(img_display, f"GRAVANDO: {frame_count} frames", (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                # Indicador piscante de gravação
                if int(time.time() * 2) % 2 == 0:
                    cv2.circle(img_display, (25, 150), 10, (0, 0, 255), -1)
            
            # Status da navegação
            if paused:
                cv2.putText(img_display, "NAVEGAÇÃO PAUSADA (pressione 'R' para continuar)", 
                           (10, HEIGHT - 40), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            
            # Fator de velocidade
            cv2.putText(img_display, f"Fator de velocidade: {speed_factor:.2f}", 
                       (WIDTH - 200, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
            
            # Comandos
            cv2.putText(img_display, "ENTER: Capturar | T: Nova sessão | +/-: Velocidade | ESC: Sair", 
                       (10, HEIGHT - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            cv2.imshow('CARLA - Navegação', img_display)
            
            # Processamento de teclas
            key = cv2.waitKey(1) & 0xFF
            
            if key == 27:  # ESC
                running = False
                print("Navegação interrompida pelo usuário.")
                return False
            elif key == 13:  # ENTER
                # Captura frame para o dataset
                if not collecting_dataset:
                    toggle_dataset_collection()
                
                if save_frame_to_dataset(current_image.copy(), current_steering):
                    print(f"Frame capturado! Total: {frame_count}")
            elif key == ord('t'):
                # Cria nova sessão do dataset
                if collecting_dataset:
                    toggle_dataset_collection()
                toggle_dataset_collection()
            elif key == ord('p') or key == ord('r'):
                # Pausa/retoma navegação
                paused = not paused if key == ord('p') else False
                print(f"Navegação {'pausada' if paused else 'retomada'}.")
                if paused:
                    # Para o veículo
                    vehicle_control.throttle = 0.0
                    vehicle_control.brake = 1.0
                    vehicle.apply_control(vehicle_control)
            elif key == ord('+') or key == ord('='):
                # Aumenta velocidade
                speed_factor = min(1.0, speed_factor + 0.05)
                print(f"Velocidade aumentada para: {speed_factor:.2f}")
            elif key == ord('-') or key == ord('_'):
                # Diminui velocidade
                speed_factor = max(0.1, speed_factor - 0.05)
                print(f"Velocidade diminuída para: {speed_factor:.2f}")
    
    # Para o veículo no final da rota
    vehicle_control.throttle = 0.0
    vehicle_control.brake = 1.0
    vehicle.apply_control(vehicle_control)
    time.sleep(1)  # Deixa o veículo parar completamente
    
    print("Veículo parado.")
    return True  # Rota completada normalmente

def main():
    parser = argparse.ArgumentParser(description='CARLA Coleta de Dados Contínua com Auto-Reverso')
    parser.add_argument('--map', type=str, default='Town01_Opt', help='Mapa a usar Town05')
    parser.add_argument('--speed', type=float, default=0.2, help='Fator de velocidade inicial (0.1-1.0)')
    parser.add_argument('--loops', type=int, default=0, help='Número de loops para dirigir (0 = infinito)')
    args = parser.parse_args()
    
    client = carla.Client('localhost', 2000)
    client.set_timeout(10.0)
 
    # try:
    #     world = client.load_world(args.map)
    #     print(f"Mapa carregado: {args.map}")
    # except Exception as e:
    #     print(f"Erro ao carregar mapa: {e}")
    #     world = client.get_world()
    #     print(f"Usando mapa atual: {world.get_map().name}")
    world = client.get_world()
    
    # Reduz complexidade de renderização para melhorar desempenho
    world.unload_map_layer(carla.MapLayer.All)
    #world.load_map_layer(carla.MapLayer.Ground)
    world.unload_map_layer(carla.MapLayer.Decals)
    world.unload_map_layer(carla.MapLayer.Props)
    world.unload_map_layer(carla.MapLayer.StreetLights)
    world.unload_map_layer(carla.MapLayer.Foliage)
    world.unload_map_layer(carla.MapLayer.ParkedVehicles)
    world.unload_map_layer(carla.MapLayer.Particles)
    world.unload_map_layer(carla.MapLayer.Walls)
    world.unload_map_layer(carla.MapLayer.Buildings)

    
    # Ativa modo síncrono
    settings = world.get_settings()
    original_settings = settings
    settings.synchronous_mode = True
    settings.fixed_delta_seconds = 0.05
    world.apply_settings(settings)
    print("Modo síncrono ativado")
    
    actors_list = []
    
    try:
        # Seleciona blueprint para o veículo
        blueprint_library = world.get_blueprint_library()
        vehicle_bp = blueprint_library.find('vehicle.tesla.model3')
        
        # Obtém pontos de spawn
        spawn_points = world.get_map().get_spawn_points()
        if not spawn_points:
            print("Nenhum ponto de spawn disponível!")
            return
        
        # Escolhe um ponto de spawn
        spawn_point = random.choice(spawn_points)
        print(f"Ponto de spawn selecionado: {spawn_point}")
        
        # Cria o veículo
        print("Criando veículo...")
        vehicle = world.spawn_actor(vehicle_bp, spawn_point)
        actors_list.append(vehicle)
        print(f"Veículo criado: {vehicle.type_id}")
        
        # Configura câmera
        camera_bp = blueprint_library.find('sensor.camera.rgb')
        camera_bp.set_attribute('image_size_x', str(WIDTH))
        camera_bp.set_attribute('image_size_y', str(HEIGHT))
        camera_bp.set_attribute('fov', '90')
        
        camera_transform = carla.Transform(
            carla.Location(x=2.0, z=1.4),
            carla.Rotation(pitch=-15)
        )
        
        camera = world.spawn_actor(camera_bp, camera_transform, attach_to=vehicle)
        actors_list.append(camera)
        print("Câmera criada")
        
        # Registra função de callback para a câmera
        camera.listen(process_image)
        
        # Cria janela para exibição
        cv2.namedWindow('CARLA - Navegação', cv2.WINDOW_NORMAL)
        

        # Loop principal para direção contínua com alternância de direção
        direction_forward = True
        current_speed = args.speed
        loop_count = 0
        
        # Encontra a rota mais longa
        print("Procura a rota mais longa no mapa...")
        route, end_point = get_longest_route(world, spawn_point, spawn_points)
        
        # Configura rota inicial
        if not route or len(route) < 10:
            print("Não foi possível encontrar uma rota adequada. Tente outro ponto de spawn.")
            return
        
 
        for i, wp in enumerate(route):
            if i % 10 == 0:  # Desenha apenas 10% dos waypoints para não sobrecarregar
                world.debug.draw_string(wp[0].transform.location, 'O', draw_shadow=False,
                                       color=carla.Color(r=0, g=0, b=255), life_time=120.0)
        
        # Loop de direção contínua
        while True:
            # Dirige a rota atual
            print(f"\n=== DIREÇÃO {'NORMAL' if direction_forward else 'REVERSA'} (Loop {loop_count+1}) ===")
            
            # Esta função retorna False se o usuário pressionou ESC
            if not drive_route(world, vehicle, route, direction_forward, current_speed):
                break
            
            # Alterna direção para próxima iteração
            direction_forward = not direction_forward
            loop_count += 0.5  # Conta um loop completo como ida + volta
            
            print(f"\nRota completada! Alternando automaticamente para direção {('NORMAL' if direction_forward else 'REVERSA')}")
            
            # Verifica se atingimos o limite de loops
            if args.loops > 0 and loop_count >= args.loops:
                print(f"\nCompletamos {args.loops} loops conforme solicitado.")
                break
    
    finally:
        # Restaura configurações originais do mundo
        world.apply_settings(original_settings)
        
        # Limpa atores
        print("Limpando atores...")
        for actor in actors_list:
            if actor is not None and actor.is_alive:
                actor.destroy()
        
        # Fecha arquivo do dataset se estiver aberto
        if 'dataset_file' in globals() and dataset_file:
            dataset_file.close()
            print(f"Dataset salvo com {frame_count} frames")
        
        cv2.destroyAllWindows()
        print("Simulação terminada")

if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print("Execução interrompida pelo usuário")
        cv2.destroyAllWindows()
    except Exception as e:
        print(f"Erro: {e}")
        cv2.destroyAllWindows()