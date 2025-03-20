import carla
import cv2
import numpy as np
import torch
import time
import carla
import cv2
import numpy as np
import time
import random
import sys

from app_training_torch import NvidiaModel, img_preprocess

# Configurações
LARGURA = 640
ALTURA = 480
MODEL_PATH = 'melhor_modelo.pth'


def main():
 
    client = carla.Client('localhost', 2000)
    client.set_timeout(10.0)
    
    modelo = NvidiaModel()
    modelo.load_state_dict(torch.load(MODEL_PATH, map_location='cpu'))
    modelo.eval()
    print(f"Modelo carregado de {MODEL_PATH}")
 
    world = client.get_world()
    print(f"A usar mapa: {world.get_map().name}")

    available_maps = client.get_available_maps()
    print("Mapas disponíveis:")
    for map_name in available_maps:
        print(f"  - {map_name}")
    
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
        ponto_spawn=carla.Transform(carla.Location(x=408.696899, y=-32.305439, z=0.281942), carla.Rotation(pitch=0.000000, yaw=-89.560913, roll=0.000000))

        
        
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
        autopilot_enabled = False
        traffic_manager = client.get_trafficmanager()
        traffic_manager.set_synchronous_mode(modo_sincrono)
        traffic_manager.set_global_distance_to_leading_vehicle(2.5)
        # Isto faz com que o veículo siga as regras de trânsito
        # Valores mais baixos são mais agressivos
        traffic_manager.global_percentage_speed_difference(10.0)
        
        #  janela única para mostrar tudo
        cv2.namedWindow('CARLA', cv2.WINDOW_NORMAL)
      
 
   
        
 
        # Função para processar a imagem da câmara
        def processar_imagem(imagem):
            nonlocal imagem_atual 
 
            array = np.frombuffer(imagem.raw_data, dtype=np.dtype("uint8"))
            array = np.reshape(array, (imagem.height, imagem.width, 4))
            imagem_atual = array[:, :, :3].copy()
            
           
        
        # Registar a função de callback
        camera.listen(processar_imagem)
        
        # Controlos do carro
        controlo_atual = carla.VehicleControl()
        
        # Ciclo principal
        a_correr = True
        print("Controlos: W/S - Acelerar/Travar | A/D - Virar | ESPAÇO - Travão de mão | ESC - Sair")
        
        while a_correr:
            if modo_sincrono:
                world.tick()
            else:
                time.sleep(0.05)  #  não sobrecarregar o CPU
            
            controlo_atual = veiculo.get_control()
            steering_atual = controlo_atual.steer
            if imagem_atual is not None:
                
                
                # Converter imagem original para BGR 
                img_original =imagem_atual.copy()# cv2.cvtColor(imagem_atual, cv2.COLOR_RGB2BGR)

                #img_original = imagem_atual

                img_rgb =imagem_atual.copy()# cv2.cvtColor(imagem_atual, cv2.COLOR_BGR2RGB)
                img_processada = img_preprocess(img_rgb)
                tensor = torch.FloatTensor(img_processada).permute(2, 0, 1).unsqueeze(0)
                with torch.no_grad():
                    angulo = -modelo(tensor).item()
                
                # Mostrar predição
                velocidade = veiculo.get_velocity()
                rapidez = 3.6 * np.sqrt(velocidade.x**2 + velocidade.y**2 + velocidade.z**2)  # km/h
                info = f"Vel: {rapidez:.1f} km/h | Steer: {steering_atual:.2f}"

                erro = angulo - steering_atual
                
                cv2.putText(img_original, f"Predict: {angulo:.3f} | Erro: {erro:.3f}", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

                cv2.putText(img_original, info, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.putText(img_original, "ESC: Sair | W/S: Acel/Trav | A/D: Virar", (10, ALTURA - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                
                
               
                
                cv2.imshow('CARLA', img_original)

         
         
                
        
                key = cv2.waitKey(1) & 0xFF
                
       
                if key == 27:  # ESC
                    a_correr = False
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
                elif key == ord('p'):  #  ativar/desativar piloto automático
    
                    autopilot_enabled = not autopilot_enabled
                    controlo_atual.throttle = 0.25
                    controlo_atual.brake = 0.0
                    veiculo.apply_control(controlo_atual)
                    veiculo.set_autopilot(autopilot_enabled)
                    print(f"Piloto automático {'ativado' if autopilot_enabled else 'desativado'}")
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