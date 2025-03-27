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

 
LIGHT_GROUP = {
    'None' : [carla.LightGroup.NONE],
    # 'Vehicle' : [carla.LightGroup.Vehicle],
    'Street' : [carla.LightGroup.Street],
    'Building' : [carla.LightGroup.Building],
    'Other' : [carla.LightGroup.Other]}

 
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




def process_image(image):
    """Função de callback para processar imagens da câmera"""
    global current_image
    array = np.frombuffer(image.raw_data, dtype=np.dtype("uint8"))
    array = np.reshape(array, (image.height, image.width, 4))
    current_image = array[:, :, :3].copy()

def simple_navigate(world, vehicle,   speed_factor=0.5):
 
    global current_image
    
    # Configurações iniciais
    vehicle_control = carla.VehicleControl()
    
    # Loop principal
    running = True
    paused = False
    steering_angle = 0.0
    start_time = time.time()
    auto_pilot = False
    
 
    print("ENTER: Capturar frame | P: Pausar | +/-: Velocidade | ESC: Sair")
    print("A/D: Virar esquerda/direita | S: Travar | W: Acelerar | Espaço: Centralizar direção | G : Autopilot")
    
    while running :
        world.tick()
        
        if not paused:
            vehicle_control.throttle = speed_factor
            vehicle_control.steer = steering_angle
            vehicle_control.brake = 0.0
            vehicle.apply_control(vehicle_control)

        controlo_atual = vehicle.get_control()
        steering_atual = controlo_atual.steer
        
        # Mostra imagens
        if current_image is not None:
            img_display = current_image.copy()
            
 
            velocity = vehicle.get_velocity()
            speed = 3.6 * np.sqrt(velocity.x**2 + velocity.y**2 + velocity.z**2)  # km/h
           
            



            # Informações básicas
            cv2.putText(img_display, f"Speed: {speed:.1f} km/h", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(img_display, f"Direction: {steering_atual:.2f}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    
            
            # Informações do dataset
            if collecting_dataset:
                cv2.putText(img_display, f"GRAVANDO: {frame_count} frames", (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                if int(time.time() * 2) % 2 == 0:
                    cv2.circle(img_display, (25, 150), 10, (0, 0, 255), -1)
            
            cv2.putText(img_display, f"Fator: {speed_factor:.2f}", (WIDTH - 200, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
            cv2.putText(img_display, "ENTER: Capturar | T: Nova sessão | A/D: Direção | ESC: Sair", (10, HEIGHT - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            cv2.imshow('CARLA', img_display)
            
            key = cv2.waitKey(1) & 0xFF
            
            if key == 27:  # ESC
                running = False
                vehicle_control = carla.VehicleControl()
                return False
            elif key == 13:  # ENTER
                if not collecting_dataset:
                    toggle_dataset_collection()
                
                if save_frame_to_dataset(current_image.copy(), steering_atual):
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
            elif key == ord('a'):
                # Vira à esquerda
                steering_angle = max(-1.0, steering_angle - 0.05)
                print(f"Virando à esquerda: {steering_angle:.2f}")
            elif key == ord('d'):
                # Vira à direita
                steering_angle = min(1.0, steering_angle + 0.05)
                print(f"Virando à direita: {steering_angle:.2f}")
            elif key == ord('w'):
                # Aumenta aceleração temporariamente
                vehicle_control.throttle = min(1.0, speed_factor + 0.2)
                print(f"Acelerando: {vehicle_control.throttle:.2f}")
            elif key == ord('s'):
                vehicle_control.throttle = 0.0
                vehicle_control.brake = 0.8
                vehicle.apply_control(vehicle_control)
                print("Trava")
            elif key == 32:  # Espaço
                steering_angle = 0.0
                print("Direção centralizada")
            elif key == ord('g'):  
                auto_pilot = not auto_pilot
                if auto_pilot:
                    vehicle.set_autopilot(True)
                else:
                    vehicle.set_autopilot(False)
                print(f"Modo de navegação {'automatico' if auto_pilot else 'manual'}")


    
    # Para o veículo no final da navegação
    vehicle_control.throttle = 0.0
    vehicle_control.brake = 1.0
    vehicle.apply_control(vehicle_control)
    time.sleep(1)  # Deixa o veículo parar completamente
    
    print("Veículo parado.")
    return True  # Navegação completada normalmente

def setup_weather_and_time(world, hour=17, weather="Clear"):
    """
    Configura o clima e a hora do dia no CARLA.
    
    Parâmetros:
    - world: objeto mundo do CARLA
    - hour: hora do dia (0-23)
    - weather: tipo de clima ("Clear", "Cloudy", "Rain", "HardRain", "Sunset")
    """
    # Configura a hora do dia
    weather_params = world.get_weather()
    
    # Define a hora do dia (azimute solar)
    # 0 = meia-noite, 90 = meio-dia, 180 = meia-noite, 270 = meio-dia
    if hour >= 0 and hour < 12:
        # Manhã: 0h = 180°, 12h = 90°
        azimuth = 180 - (hour * 7.5)
    else:
        # Tarde/Noite: 12h = 90°, 24h = 0°
        azimuth = 90 - ((hour - 12) * 7.5)
    
    # Calcula a elevação solar (altura no céu)
    # Ao meio-dia a elevação é máxima, à meia-noite é mínima
    if hour >= 6 and hour <= 18:
        # Dia: 6h = 0°, 12h = 90°, 18h = 0°
        if hour <= 12:
            elevation = (hour - 6) * 15
        else:
            elevation = (18 - hour) * 15
    else:
        # Noite: elevação negativa (sol abaixo do horizonte)
        if hour > 18:
            elevation = -15
        else:
            elevation = -15
    
    weather_params.sun_azimuth_angle = azimuth
    weather_params.sun_altitude_angle = elevation
    
    # Configura o tipo de clima
    if weather == "Clear":
        weather_params.cloudiness = 10.0
        weather_params.precipitation = 0.0
        weather_params.precipitation_deposits = 0.0
        weather_params.wetness = 0.0
    elif weather == "Cloudy":
        weather_params.cloudiness = 80.0
        weather_params.precipitation = 0.0
        weather_params.precipitation_deposits = 0.0
        weather_params.wetness = 0.0
    elif weather == "Rain":
        weather_params.cloudiness = 80.0
        weather_params.precipitation = 60.0
        weather_params.precipitation_deposits = 40.0
        weather_params.wetness = 40.0
    elif weather == "HardRain":
        weather_params.cloudiness = 90.0
        weather_params.precipitation = 90.0
        weather_params.precipitation_deposits = 80.0
        weather_params.wetness = 80.0
    elif weather == "Sunset":
        weather_params.cloudiness = 15.0
        weather_params.precipitation = 0.0
        weather_params.precipitation_deposits = 0.0
        weather_params.wetness = 0.0
        weather_params.sun_azimuth_angle = 90.0
        weather_params.sun_altitude_angle = 15.0  # Sol baixo no horizonte
    
    # Para final de tarde (17h),  
    # aumenta a densidade de neblina e a distância da neblina
    if hour >= 17 and hour <= 19:
        weather_params.fog_density = 10.0
        weather_params.fog_distance = 75.0
        weather_params.fog_falloff = 1.0
        
    world.set_weather(weather_params)
    print(f"Clima configurado: {weather}, Hora: {hour}:00")

def apply_lights_manager( light_manager):
 

    light_group = 'Street'

    # filter by group
    lights = light_manager.get_all_lights(LIGHT_GROUP[light_group][0]) # light_group
    light_manager.turn_off(lights)

    # i = 0
    # while (i < len(args.lights)):
    #     option = args.lights[i]

    #     if option == "on":
    #         light_manager.turn_on(lights)
    #     elif option == "off":
    #         light_manager.turn_off(lights)
    #     elif option == "intensity":
    #         light_manager.set_intensity(lights, int(args.lights[i + 1]))
    #         i += 1
    #     elif option == "color":
    #         r = int(args.lights[i + 1])
    #         g = int(args.lights[i + 2])
    #         b = int(args.lights[i + 3])
    #         light_manager.set_color(lights, carla.Color(r, g, b))
    #         i += 3

    #     i += 1

def main():
    parser = argparse.ArgumentParser(description='CARLA')
    parser.add_argument('--speed', type=float, default=0.0, help='Fator de velocidade inicial (0.1-1.0)')
 
    args = parser.parse_args()
    
    print("Conectando ao servidor CARLA...")
    client = carla.Client('localhost', 2000)
    if client is None:
        print("Erro ao conectar ao servidor CARLA.")
        return
    client.set_timeout(10.0)
   
    print("Carregando mapa...")
    world = client.get_world()
    #manager = world.get_lightmanager()
    #apply_lights_manager(manager)
 
    setup_weather_and_time(world, hour=17, weather="Clear")
    traffic_lights = world.get_actors().filter('traffic.traffic_light')

    for traffic_light in traffic_lights:
        traffic_light.set_state(carla.TrafficLightState.Off)

    world.unload_map_layer(carla.MapLayer.All)
    
 
    
    # Ativa modo síncrono
    settings = world.get_settings()
    original_settings = settings
    settings.synchronous_mode = True
    settings.fixed_delta_seconds = 0.05
    world.apply_settings(settings)
    
    
    actors_list = []
    
    try:
        print("Blue Print")
        blueprint_library = world.get_blueprint_library()
        vehicle_bp = blueprint_library.find('vehicle.tesla.model3')
        
        # Obtém pontos de spawn
        print("Obtendo pontos de spawn...")
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
        print("Criando câmera...")
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
        
 
        camera.listen(process_image)
        
 
        
 
        
        simple_navigate(world, vehicle,   speed_factor=args.speed)
    
    finally:
        camera.stop()
        settings = world.get_settings()
        settings.synchronous_mode = False
        settings.fixed_delta_seconds = None
        world.apply_settings(settings)
    
        if 'lista_atores' in locals():
            print("A limpar atores...")
            for ator in actors_list:
                if ator is not None and ator.is_alive:
                    ator.destroy()

        
 
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
