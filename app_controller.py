import carla
import cv2
import numpy as np
import time
import random
import sys
import os
import datetime
 
is_recording = False
video_writer = None
recording_start_time = None

collecting_dataset = False
dataset_dir = None
dataset_images_dir = None
dataset_file = None
frame_count = 0

def create_dataset_session( ):
    global dataset_dir, dataset_images_dir , frame_count, dataset_file
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
    global collecting_dataset, dataset_file
    if not collecting_dataset:
        os.makedirs("dataset", exist_ok=True)
        create_dataset_session()
        collecting_dataset = True
        print("Iniciando coleta de dados para treino")
    else:
        if dataset_file:
            dataset_file.close()
        collecting_dataset = False
        print(f"Coleta de dados finalizada. Total de frames: {frame_count}")
        frame_count = 0
        
def save_frame_to_dataset(frame,steering):
    global dataset_file, frame_count
    if not dataset_file:
        print("Coleta de dados desligada.")
        return
        
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    image_filename = f"frame_{timestamp}.jpg"
    image_path = f"{dataset_images_dir}/{image_filename}"
    print(f"Salvando frame para dataset: {image_path}")
    cv2.imwrite(image_path, frame)
    
    dataset_file.write(f"images/{image_filename},{steering:.6f}\n")
    dataset_file.flush()
    
    frame_count += 1
    
    if frame_count % 10 == 0:
        print(f"Frames capturados: {frame_count}", end="\r")

    
 
    

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
    

#   - /Game/Carla/Maps/Town01_Opt
#   - /Game/Carla/Maps/Town03
#   - /Game/Carla/Maps/Town04_Opt
#   - /Game/Carla/Maps/Town01
#   - /Game/Carla/Maps/Town05
#   - /Game/Carla/Maps/Town03_Opt
#   - /Game/Carla/Maps/Town04
#   - /Game/Carla/Maps/Town10HD_Opt
#   - /Game/Carla/Maps/Town05_Opt
#   - /Game/Carla/Maps/Town02
#   - /Game/Carla/Maps/Town10HD
#   - /Game/Carla/Maps/Town02_Opt



    # try:
    #     world = client.load_world('Town04_Opt')
        
    #     # if '/Game/Carla/Maps/Town04_Opt' in available_maps:
    #     #     world = client.load_world('Town04_Opt')
    #     #     print("Carregado mapa leve: Town04_Opt")
    #     # # Town05 que é mais aberto e geralmente mais leve
    #     # elif '/Game/Carla/Maps/Town05' in available_maps:
    #     #     world = client.load_world('Town05')
    #     #     print("Carregado mapa leve: Town05")
    #     # else:
    #     #     world = client.get_world()
    #     #     print(f"Usando mapa atual: {world.get_map().name}")
    # except Exception as e:
    #     print(f"Erro ao carregar mapa: {e}")
    #     world = client.get_world()
    #     print(f"Usando mapa atual: {world.get_map().name}")
    

        world.unload_map_layer(carla.MapLayer.All)
        # world.load_map_layer(carla.MapLayer.Ground)
        # world.load_map_layer(carla.MapLayer.Buildings)
        # world.load_map_layer(carla.MapLayer.Props)
        # world.load_map_layer(carla.MapLayer.StreetLights)
        

        world.unload_map_layer(carla.MapLayer.Decals)
        world.unload_map_layer(carla.MapLayer.Props)
        world.unload_map_layer(carla.MapLayer.StreetLights)
        world.unload_map_layer(carla.MapLayer.Foliage)
        world.unload_map_layer(carla.MapLayer.ParkedVehicles)
        world.unload_map_layer(carla.MapLayer.Particles)
        world.unload_map_layer(carla.MapLayer.Walls)
        
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
     

        #create_dataset_session()
 
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
        #traffic_manager.global_percentage_speed_difference(50)
        traffic_manager.set_global_distance_to_leading_vehicle(2.5)
        world.constant_velocity_enabled = True
        # Isto faz com que o veículo siga as regras de trânsito
        # Valores mais baixos são mais agressivos
        traffic_manager.global_percentage_speed_difference(10.0)
        a_correr = True
        print("Controlos: W/S - Acelerar/Travar | A/D - Virar | ESPAÇO - Travão de mão | ESC - Sair")
        is_steering = False 
        while a_correr:
            if modo_sincrono:
                world.tick()
            else:
                time.sleep(0.05)  #  não sobrecarregar o CPU
            
            controlo_atual = veiculo.get_control()
            steering_atual = controlo_atual.steer
                
            
            # Mostrar as imagens
            if imagem_atual is not None:
          
                
                # Converter imagem original para BGR 
                frame =imagem_atual.copy()
                img_original =imagem_atual.copy()# cv2.cvtColor(imagem_atual, cv2.COLOR_RGB2BGR)


 
            
                
                # Adicionar informações à imagem original
                velocidade = veiculo.get_velocity()
                rapidez = 3.6 * np.sqrt(velocidade.x**2 + velocidade.y**2 + velocidade.z**2)  # km/h
                info = f"Vel: {rapidez:.1f} km/h | Steer: {steering_atual:.2f}"
                cv2.putText(img_original, info, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                if not autopilot_enabled:
                    cv2.putText(img_original, "ESC: Sair | W/S: Acel/Trav | A/D: Virar", (10, ALTURA - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                else:
                    cv2.putText(img_original, "ESC: Sair | Autopilot", (10, ALTURA - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

                
                font = cv2.FONT_HERSHEY_SIMPLEX
                font_scale = 0.6
                font_thickness = 2
                if collecting_dataset:
                    dataset_text = f"DATASET: {frame_count} frames"
                    if int(time.time() * 2) % 2 == 0:
                        cv2.circle(img_original, (30, 130), 10, (255, 0, 0), -1)
                    cv2.putText(img_original, dataset_text, (45, 135), font, font_scale, (255, 0, 0), font_thickness)
                    cv2.putText(img_original, "ENTER  capturar frame", (10, 165), font, font_scale, (255, 255, 0), font_thickness)
        
            
                
              

                
             
                cv2.imshow('CARLA', img_original)

            
 
        
                key = cv2.waitKey(1) & 0xFF
                #controlo_atual.steer *= 0.99
                

       
                if key == 27:  # ESC
                    a_correr = False
                if key == 13:  # Enter key
                    save_frame_to_dataset(frame,steering_atual)
                if key == ord('t'):
                    toggle_dataset_collection()
                if key == ord('p'):  #   ativar/desativar piloto automático
 
                    autopilot_enabled = not autopilot_enabled
                    #controlo_atual.throttle = 0.15
                    #controlo_atual.brake = 0.0
                    #veiculo.apply_control(controlo_atual)
                    veiculo.set_autopilot(autopilot_enabled)
                    print(f"Piloto automático {'ativado' if autopilot_enabled else 'desativado'}")
             
                if key == ord('w'):
                    controlo_atual.throttle = 0.2
                    controlo_atual.brake = 0.0
                    controlo_atual.gear = 1
                elif key == ord('s'):
                    controlo_atual.throttle = 0.1
                    controlo_atual.gear = -1
                
                if key == ord('a'):
                    controlo_atual.steer += -0.05
                    is_steering = True
                elif key == ord('d'):
                    controlo_atual.steer += 0.05
                    is_steering = True 
                else:
                    is_steering = False
                    if not autopilot_enabled:
                        controlo_atual.steer *= 0.99 
                
                
                if key == ord(' '):  # spce
                    controlo_atual.brake = 0.9
                    controlo_atual.throttle = 0.0
                    controlo_atual.hand_brake = not controlo_atual.hand_brake
                # else:
 
                #     if key not in [ord('w'), ord('s')]:
                #         controlo_atual.throttle = 0.0
                #         controlo_atual.brake = 0.0

                if controlo_atual.steer > 1.0:
                    controlo_atual.steer = 1.0
                elif controlo_atual.steer < -1.0:
                    controlo_atual.steer = -1.0
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
        
        if collecting_dataset and dataset_file:
            dataset_file.close()
            print(f"Dataset salvo com {frame_count} frames")

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