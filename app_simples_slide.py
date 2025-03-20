import carla
import cv2
import numpy as np
import time
import random
import sys

 

# Configurações
LARGURA = 640
ALTURA = 480

 


thresh_value = 200
canny_min = 50
canny_max = 150
hough_threshold = 50
min_line_length = 50
max_line_gap = 100
inclinacao_max = 50  # será dividido por 100 (0.5)

def nothing(x):
    pass

def detetar_linhas_estrada(imagem, params):
 
    thresh_val, canny_min, canny_max, hough_thresh, min_line_len, max_gap, inclinacao_max = params
    inclinacao_max = inclinacao_max / 100.0  # Converter para decimal (0-1)
    
    # Converter para escala de cinzentos
    cinza = cv2.cvtColor(imagem, cv2.COLOR_BGR2GRAY)
    
    # Definir região de interesse (apenas parte inferior da imagem)
    altura, largura = cinza.shape
    mascara = np.zeros_like(cinza)
    
    # Preencher o triângulo inferior da imagem
    pontos_roi = np.array([[(0, altura), (0, altura//2), (largura, altura//2), (largura, altura)]], dtype=np.int32)
    cv2.fillPoly(mascara, pontos_roi, 255)
    imagem_roi = cv2.bitwise_and(cinza, mascara)
    
    # Destacar linhas brancas
    ret, binario = cv2.threshold(imagem_roi, thresh_val, 255, cv2.THRESH_BINARY)
    
    # Deteção de bordas
    bordas = cv2.Canny(binario, canny_min, canny_max)
    
    # Criar imagem colorida para mostrar as bordas
    bordas_coloridas = cv2.cvtColor(bordas, cv2.COLOR_GRAY2BGR)
    
    # Deteção de linhas com transformada de Hough
    linhas = cv2.HoughLinesP(
        bordas,
        rho=1,
        theta=np.pi/180,
        threshold=hough_thresh,
        minLineLength=min_line_len,
        maxLineGap=max_gap
    )
    
    # Criar imagem para mostrar as linhas detetadas
    imagem_linhas = np.zeros_like(imagem)
    linhas_esquerda = []
    linhas_direita = []
    
    if linhas is not None:
        for linha in linhas:
            x1, y1, x2, y2 = linha[0]
            
            # Calcular inclinação para determinar se é linha esquerda ou direita
            if x2 - x1 == 0:  # Evitar divisão por zero
                inclinacao = 999  # Valor arbitrário grande
            else:
                inclinacao = float(y2 - y1) / float(x2 - x1)
            
            # Filtrar linhas pela inclinação e posição
            if abs(inclinacao) < inclinacao_max:  # Ignorar linhas muito horizontais
                continue
                
            centro_x = float(largura/2)
            x1_float = float(x1)
                
            if inclinacao < 0 and x1_float < centro_x:
                linhas_esquerda.append(linha)
                cv2.line(imagem_linhas, (x1, y1), (x2, y2), (255, 0, 0), 2)  # Linha azul à esquerda
            elif inclinacao > 0 and x1_float > centro_x:
                linhas_direita.append(linha)
                cv2.line(imagem_linhas, (x1, y1), (x2, y2), (0, 0, 255), 2)  # Linha vermelha à direita
    
 
    resultado = cv2.addWeighted(imagem, 0.8, imagem_linhas, 1, 0)
    
    return resultado, bordas_coloridas, binario, imagem_linhas
 


def nothing(x):
    pass

def main():
 
    client = carla.Client('localhost', 2000)
    client.set_timeout(10.0)
    
 
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
        cv2.namedWindow('Controlos')
        cv2.resizeWindow('Controlos', 600, 300)

        cv2.createTrackbar('Threshold', 'Controlos', thresh_value, 255, nothing)
        cv2.createTrackbar('Canny Min', 'Controlos', canny_min, 255, nothing)
        cv2.createTrackbar('Canny Max', 'Controlos', canny_max, 255, nothing)
        cv2.createTrackbar('Hough Thresh', 'Controlos', hough_threshold, 200, nothing)
        cv2.createTrackbar('Min Line Len', 'Controlos', min_line_length, 200, nothing)
        cv2.createTrackbar('Max Line Gap', 'Controlos', max_line_gap, 200, nothing)
        cv2.createTrackbar('Inclinação Máx', 'Controlos', inclinacao_max, 100, nothing)



        
 
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
        left_base_prev = None
        right_base_prev = None
        while a_correr:
            if modo_sincrono:
                world.tick()
            else:
                time.sleep(0.05)  #  não sobrecarregar o CPU
            
            controlo_atual = veiculo.get_control()
            steering_atual = controlo_atual.steer
            if imagem_atual is not None:
                
                
                # Converter imagem original para BGR 
                img_original =imagem_atual# cv2.cvtColor(imagem_atual, cv2.COLOR_RGB2BGR)
                imagem_bgr = cv2.cvtColor(imagem_atual, cv2.COLOR_RGB2BGR)
                
                # Adicionar informações à imagem original
                velocidade = veiculo.get_velocity()
                rapidez = 3.6 * np.sqrt(velocidade.x**2 + velocidade.y**2 + velocidade.z**2)  # km/h
                info = f"Vel: {rapidez:.1f} km/h | Steer: {steering_atual:.2f}"
                cv2.putText(img_original, info, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.putText(img_original, "ESC: Sair | W/S: Acel/Trav | A/D: Virar", (10, ALTURA - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                
                thresh_val = cv2.getTrackbarPos('Threshold', 'Controlos')
                canny_min_val = cv2.getTrackbarPos('Canny Min', 'Controlos')
                canny_max_val = cv2.getTrackbarPos('Canny Max', 'Controlos')
                hough_thresh_val = cv2.getTrackbarPos('Hough Thresh', 'Controlos')
                min_line_len_val = cv2.getTrackbarPos('Min Line Len', 'Controlos')
                max_line_gap_val = cv2.getTrackbarPos('Max Line Gap', 'Controlos')
                inclinacao_max_val = cv2.getTrackbarPos('Inclinação Máx', 'Controlos')
                
                # Empacotar parâmetros
                params = (thresh_val, canny_min_val, canny_max_val, hough_thresh_val, 
                        min_line_len_val, max_line_gap_val, inclinacao_max_val)
            
                resultado, bordas, binario, linhas = detetar_linhas_estrada(imagem_bgr, params)
                    
                # Converter binário para RGB para visualização
                #binario_vis = cv2.cvtColor(binario, cv2.COLOR_GRAY2BGR)
                
                # Mostrar o resultado
                topo = np.hstack((imagem_bgr, resultado))
                baixo = np.hstack((bordas, linhas))
                
                # Redimensionar se necessário para igualar dimensões
                if baixo.shape[1] != topo.shape[1]:
                    baixo = cv2.resize(baixo, (topo.shape[1], baixo.shape[0]))
                
     
                visualizacao = np.vstack((topo, baixo))
                
       
                info_text = f"Threshold: {thresh_val} | Canny: {canny_min_val}-{canny_max_val} | Hough: {hough_thresh_val}"
                cv2.putText(visualizacao, info_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                
                cv2.imshow('CARLA', img_original)
                cv2.imshow('DEBUG', visualizacao)
 
    
 
                imagem_bgr = cv2.cvtColor(imagem_atual, cv2.COLOR_RGB2BGR)
            
              
        
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
                    # Alterna o estado do piloto automático
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