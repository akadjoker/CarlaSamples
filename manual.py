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
    
    print("\n=== NAVEGAÇÃO SIMPLES INICIADA ===")
    print("ENTER: Capturar frame | P: Pausar | +/-: Velocidade | ESC: Sair")
    print("A/D: Virar esquerda/direita | S: Travar | W: Acelerar | Espaço: Centralizar direção")
    
    while running :
        world.tick()
        
        if not paused:
            vehicle_control.throttle = speed_factor
            vehicle_control.steer = steering_angle
            vehicle_control.brake = 0.0
            vehicle.apply_control(vehicle_control)
        
        # Mostra imagens
        if current_image is not None:
            img_display = current_image.copy()
            
 
            velocity = vehicle.get_velocity()
            speed = 3.6 * np.sqrt(velocity.x**2 + velocity.y**2 + velocity.z**2)  # km/h
            current_steering = vehicle_control.steer
            
            # Informações básicas
            cv2.putText(img_display, f"Speed: {speed:.1f} km/h", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(img_display, f"Direction: {current_steering:.2f}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    
            
            # Informações do dataset
            if collecting_dataset:
                cv2.putText(img_display, f"GRAVANDO: {frame_count} frames", (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                # Indicador piscante de gravação
                if int(time.time() * 2) % 2 == 0:
                    cv2.circle(img_display, (25, 150), 10, (0, 0, 255), -1)
            
            cv2.putText(img_display, f"Fator: {speed_factor:.2f}", 
                       (WIDTH - 200, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
            
            
            cv2.putText(img_display, "ENTER: Capturar | T: Nova sessão | A/D: Direção | ESC: Sair", 
                       (10, HEIGHT - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            cv2.imshow('CARLA', img_display)
            
            # Processamento de teclas
            key = cv2.waitKey(1) & 0xFF
            
            if key == 27:  # ESC
                running = False
                print("Navegação interrompida pelo usuário.")
                return False
            elif key == 13:  # ENTER
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
    
    # Para o veículo no final da navegação
    vehicle_control.throttle = 0.0
    vehicle_control.brake = 1.0
    vehicle.apply_control(vehicle_control)
    time.sleep(1)  # Deixa o veículo parar completamente
    
    print("Veículo parado.")
    return True  # Navegação completada normalmente

def main():
    parser = argparse.ArgumentParser(description='CARLA')
    parser.add_argument('--speed', type=float, default=0.0, help='Fator de velocidade inicial (0.1-1.0)')
 
    args = parser.parse_args()
    
    client = carla.Client('localhost', 2000)
    client.set_timeout(10.0)
 
 
    world = client.get_world()
    
 
    
    # Ativa modo síncrono
    #settings = world.get_settings()
    #original_settings = settings
    #settings.synchronous_mode = True
    #settings.fixed_delta_seconds = 0.05
    #world.apply_settings(settings)
    
    
    actors_list = []
    
    try:
   
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
        
 
        camera.listen(process_image)
        
 
        
 
        
        simple_navigate(world, vehicle,   speed_factor=args.speed)
    
    finally:
        for actor in actors_list:
            if actor is not None and actor.is_alive:
                actor.destroy()
        
 
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
