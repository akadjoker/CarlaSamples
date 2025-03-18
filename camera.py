import carla
import pygame
import numpy as np
import cv2
import time
import random
import sys

LARGURA = 800
ALTURA = 600
LARGURA_FAIXA = 400
ALTURA_FAIXA = 200
FPS = 30

def main():

    pygame.init()
    pygame.display.set_caption("CARLA - Camera Jetson")
    display = pygame.display.set_mode((LARGURA, ALTURA), pygame.HWSURFACE | pygame.DOUBLEBUF)
    pygame.event.set_allowed([pygame.QUIT, pygame.KEYDOWN, pygame.KEYUP])
    clock = pygame.time.Clock()
    
    try:

        client = carla.Client('localhost', 2000)
        client.set_timeout(10.0)
        

        world = client.get_world()

        available_maps = client.get_available_maps()
        print("Mapas disponíveis:")
        for map_name in available_maps:
            print(f"  - {map_name}")
        
        try:
            if '/Game/Carla/Maps/Town04_Opt' in available_maps:
                world = client.load_world('Town04_Opt')
                print("Carregado mapa leve: Town04_Opt")
            # Town05 que é mais aberto e geralmente mais leve
            elif '/Game/Carla/Maps/Town05' in available_maps:
                world = client.load_world('Town05')
                print("Carregado mapa leve: Town05")
            else:
                world = client.get_world()
                print(f"Usando mapa atual: {world.get_map().name}")
        except Exception as e:
            print(f"Erro ao carregar mapa: {e}")
            world = client.get_world()
            print(f"Usando mapa atual: {world.get_map().name}")
        

        world.unload_map_layer(carla.MapLayer.All)
        world.load_map_layer(carla.MapLayer.Ground)
        world.load_map_layer(carla.MapLayer.Buildings)
        world.load_map_layer(carla.MapLayer.Props)
        world.load_map_layer(carla.MapLayer.StreetLights)
        
        
        # Configura o modo síncrono para melhor desempenho
        settings = world.get_settings()
        modo_sincrono = False
        
        if not settings.synchronous_mode:
            modo_sincrono = True
            settings.synchronous_mode = True
            settings.fixed_delta_seconds = 0.05
            world.apply_settings(settings)
            print("Modo síncrono ativado")
        

        lista_atores = []
        
   
        blueprint_library = world.get_blueprint_library()
        veiculo_bp = blueprint_library.find('vehicle.tesla.model3')
        
        # Encontra um ponto de spawn
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
        
        # Configura a câmara principal (terceira pessoa)
        camera_principal_bp = blueprint_library.find('sensor.camera.rgb')
        camera_principal_bp.set_attribute('image_size_x', str(LARGURA))
        camera_principal_bp.set_attribute('image_size_y', str(ALTURA))
        camera_principal_bp.set_attribute('fov', '100')
        camera_principal_bp.set_attribute('enable_postprocess_effects', 'False')
        camera_principal_bp.set_attribute('sensor_tick', '0.1')
        
        transformacao_camera_principal = carla.Transform(
            carla.Location(x=-6.0, z=3.5),
            carla.Rotation(pitch=-15)
        )
        
        camera_principal = world.spawn_actor(camera_principal_bp, transformacao_camera_principal, attach_to=veiculo)
        lista_atores.append(camera_principal)
        print("Câmara principal criada")
        
        # Configura a câmara para deteção de linhas (visão frontal)
        camera_faixas_bp = blueprint_library.find('sensor.camera.rgb')
        camera_faixas_bp.set_attribute('image_size_x', str(LARGURA_FAIXA))
        camera_faixas_bp.set_attribute('image_size_y', str(ALTURA_FAIXA))
        camera_faixas_bp.set_attribute('fov', '90')
        camera_faixas_bp.set_attribute('enable_postprocess_effects', 'False')
        camera_faixas_bp.set_attribute('sensor_tick', '0.1')
        
        # Posiciona a câmara na frente e um pouco acima do veículo, a olhar para baixo
        transformacao_camera_faixas = carla.Transform(
            carla.Location(x=2.0, z=1.5),
            carla.Rotation(pitch=-10)
        )
        
        camera_faixas = world.spawn_actor(camera_faixas_bp, transformacao_camera_faixas, attach_to=veiculo)
        lista_atores.append(camera_faixas)
        print("Câmara de deteção de linhas criada")
        

        superficie_principal = None
        superficie_faixas = None
        imagem_detecao_faixas = None
        

        controlo_atual = carla.VehicleControl()
        
        # Função para processar a imagem da câmara principal
        def processar_imagem_principal(imagem):
            nonlocal superficie_principal
            array = np.frombuffer(imagem.raw_data, dtype=np.dtype("uint8"))
            array = np.reshape(array, (imagem.height, imagem.width, 4))
            array = array[:, :, :3]  # Descarta o canal alfa
            array = array[:, :, ::-1]  # Converte BGR para RGB
            superficie_principal = pygame.surfarray.make_surface(array.swapaxes(0, 1))
        
        # Função para processar a imagem da câmara de deteção de linhas
        def processar_imagem_faixas(imagem):
            nonlocal superficie_faixas, imagem_detecao_faixas
            array = np.frombuffer(imagem.raw_data, dtype=np.dtype("uint8"))
            array = np.reshape(array, (imagem.height, imagem.width, 4))
            array = array[:, :, :3]  # Descarta o canal alfa
            # Converter para formato OpenCV (RGB -> BGR)
            imagem_opencv = array.copy()
            # Converter novamente para RGB para mostrar no Pygame
            faixas_rgb = cv2.cvtColor(imagem_opencv, cv2.COLOR_BGR2RGB)
            # Criar superfície Pygame
            superficie_faixas = pygame.surfarray.make_surface(faixas_rgb.swapaxes(0, 1))
        
     
        
        # Configurar sensores
        camera_principal.listen(processar_imagem_principal)
        camera_faixas.listen(processar_imagem_faixas)
        
        # Iniciamos o carrito
        veiculo.apply_control(carla.VehicleControl(throttle=0.0))
        

        a_correr = True
        print("Controlos: W/S - Acelerar/Travar | A/D - Virar | ESPAÇO - Travão de mão | ESC - Sair")
        
        while a_correr:
 
            if modo_sincrono:
                world.tick()
            
            clock.tick(FPS)
            

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    a_correr = False
                
    
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        a_correr = False
     
                    if event.key == pygame.K_w:
                        controlo_atual.throttle = 0.7
                        controlo_atual.brake = 0.0
                    if event.key == pygame.K_s:
                        controlo_atual.throttle = 0.0
                        controlo_atual.brake = 0.7
                    if event.key == pygame.K_a:
                        controlo_atual.steer = -0.5
                    if event.key == pygame.K_d:
                        controlo_atual.steer = 0.5
                    if event.key == pygame.K_SPACE:
                        controlo_atual.hand_brake = True
                    
      
                    veiculo.apply_control(controlo_atual)
                
 
                if event.type == pygame.KEYUP:
                    if event.key == pygame.K_w:
                        controlo_atual.throttle = 0.0
                    if event.key == pygame.K_s:
                        controlo_atual.brake = 0.0
                    if event.key == pygame.K_a or event.key == pygame.K_d:
                        controlo_atual.steer = 0.0
                    if event.key == pygame.K_SPACE:
                        controlo_atual.hand_brake = False
                    
    
                    veiculo.apply_control(controlo_atual)
            
 
            if superficie_principal is not None and superficie_faixas is not None:
                # Mostrar a imagem principal
                display.blit(superficie_principal, (0, 0))
                
                # Mostrar a imagem de deteção de linhas no canto superior direito
                retangulo_faixas = pygame.Rect(LARGURA - LARGURA_FAIXA, 0, LARGURA_FAIXA, ALTURA_FAIXA)
                display.blit(superficie_faixas, retangulo_faixas)
                
                # Desenhar uma borda à volta da deteção de linhas
                pygame.draw.rect(display, (255, 255, 255), retangulo_faixas, 2)
                
                # Textos informativos
                fonte = pygame.font.Font(None, 28)
                texto_controlos = fonte.render("W/S: Aceleração | A/D: Direção | ESPAÇO: Travão | ESC: Sair", True, (255, 255, 255))
                display.blit(texto_controlos, (10, ALTURA - 30))
                
                # Mostrar velocidade e FPS
                velocidade = veiculo.get_velocity()
                rapidez = 3.6 * np.sqrt(velocidade.x**2 + velocidade.y**2 + velocidade.z**2)  # km/h
                fps_atual = clock.get_fps()
                texto_info = f"Vel: {rapidez:.1f} km/h | FPS: {fps_atual:.1f} | Deteção de Linhas"
                texto_velocidade = fonte.render(texto_info, True, (255, 255, 255))
                display.blit(texto_velocidade, (10, 10))
                

                pygame.display.flip()
    
    finally:
        try:
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
            
            pygame.quit()
            print("Simulação terminada")
            
        except Exception as e:
            print(f"Erro durante limpeza final: {e}")

if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print("Exit")
    except Exception as e:
        print(f"Erro: {e}")