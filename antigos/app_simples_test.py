import carla
import cv2
import numpy as np
import time
import random
import sys

from lane_detection import (
    Line, left_line, right_line, 
    find_lane_pixels, fit_polynomial, 
    apply_thresholds, perspective_transform,apply_thresholds_carla
)

# Configurações
LARGURA = 640
ALTURA = 480

threshold_min = 200
threshold_max = 255
sobel_min = 30
sobel_max = 150
canny_min = 50
canny_max = 150

def line_detection(imagem_bgr):
    # Converter de RGB para BGR para o OpenCV
        #imagem_bgr = cv2.cvtColor(imagem_atual, cv2.COLOR_RGB2BGR)
        
        try:
            src = np.float32([
                [LARGURA*0.25, ALTURA],       # Inferior esquerdo
                [LARGURA*0.45, ALTURA*0.6],   # Superior esquerdo
                [LARGURA*0.55, ALTURA*0.6],   # Superior direito
                [LARGURA*0.75, ALTURA]        # Inferior direito
            ])
            
            dst = np.float32([
                [LARGURA*0.3, ALTURA],    # Inferior esquerdo
                [LARGURA*0.3, 0],         # Superior esquerdo
                [LARGURA*0.7, 0],         # Superior direito
                [LARGURA*0.7, ALTURA]     # Inferior direito
            ])
            thresh_min = cv2.getTrackbarPos('Threshold Min', 'CARLA')
            thresh_max = cv2.getTrackbarPos('Threshold Max', 'CARLA')
            sobel_min = cv2.getTrackbarPos('Sobel Min', 'CARLA')
            sobel_max = cv2.getTrackbarPos('Sobel Max', 'CARLA')
            canny_min = cv2.getTrackbarPos('Canny Min', 'CARLA')
            canny_max = cv2.getTrackbarPos('Canny Max', 'CARLA')
            
            slider_vals = (thresh_min, thresh_max, sobel_min, sobel_max, canny_min, canny_max)
            
            # 1. Aplicar limiares adaptados para CARLA com valores dos sliders
            binario = apply_thresholds_carla(imagem_bgr, slider_vals)
            
            # 2. Aplicar transformação de perspetiva
            warped, M, Minv = perspective_transform(binario, src, dst)
            
            # 3. Ajustar polinómios às linhas
            left_fitx, right_fitx, result, out_img, ploty = fit_polynomial(warped)
            
            # 4. Criar visualização final
            if left_fitx is not None and right_fitx is not None:
                # Visualizar as faixas na imagem original
                warp_zero = np.zeros_like(warped).astype(np.uint8)
                color_warp = np.dstack((warp_zero, warp_zero, warp_zero))
                
                # Reformatar arrays para desenhar o polígono
                pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
                pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
                pts = np.hstack((pts_left, pts_right))
                
                # Desenhar o polígono na imagem
                cv2.fillPoly(color_warp, np.int32([pts]), (0, 255, 0))
                
                # Voltar à perspetiva original
                newwarp = cv2.warpPerspective(color_warp, Minv, (imagem_bgr.shape[1], imagem_bgr.shape[0]))
                
                # Combinar com a imagem original
                final_result = cv2.addWeighted(imagem_bgr, 1, newwarp, 0.3, 0)
                
                # Adicionar métricas à imagem final
                raio_medio = (left_line.radius_of_curvature + right_line.radius_of_curvature) / 2
                offset = (left_line.line_base_pos + right_line.line_base_pos) / 2
                
                info_text = f"Raio: {raio_medio:.1f}m | Offset: {offset:.2f}m"
                cv2.putText(final_result, info_text, (10, 30), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                
                # Preparar imagens para visualização
                binario_rgb = np.dstack((binario*255, binario*255, binario*255))
                warped_rgb = np.dstack((warped*255, warped*255, warped*255))
                
                # Criar grelha para visualização
                topo = np.hstack((imagem_bgr, final_result))
                meio = np.hstack((binario_rgb, warped_rgb))
                baixo = np.hstack((out_img, result))
                
                # Garantir que todas as imagens têm o mesmo tamanho
                meio = cv2.resize(meio, (topo.shape[1], meio.shape[0]))
                baixo = cv2.resize(baixo, (topo.shape[1], baixo.shape[0]))
                
                # Combinar todas as visualizações
                grelha = np.vstack((topo, meio, baixo))
                
                # Redimensionar para caber no ecrã
                escala = min(1.0, 1200 / grelha.shape[1])
                dim = (int(grelha.shape[1] * escala), int(grelha.shape[0] * escala))
                grelha_redim = cv2.resize(grelha, dim)
                
                # Mostrar grelha
                cv2.imshow('DEBUG', grelha_redim)
            else:
                # Se não detetou linhas, mostrar só imagem original e binário
                binario_rgb = np.dstack((binario*255, binario*255, binario*255))
                warped_rgb = np.dstack((warped*255, warped*255, warped*255))
                
                # Juntar as imagens
                visualizacao = np.hstack((imagem_bgr, binario_rgb))
                
                # Adicionar texto
                cv2.putText(visualizacao, "Sem Linhas", (10, 30), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                
                # Mostrar
                cv2.imshow('DEBUG', visualizacao)
        
        except Exception as e:
            print(f"Erro ao processar imagem: {e}")
            #traceback.print_exc()
            # Mostrar imagem original em caso de erro
            cv2.imshow('DEBUG', imagem_bgr)

def detetar_linhas_janela_deslizante(imagem):
    # Converter para escala de cinzentos
    cinza = cv2.cvtColor(imagem, cv2.COLOR_BGR2GRAY)
    
    # Aplicar limiar para isolar linhas brancas
    _, binario = cv2.threshold(cinza, 200, 255, cv2.THRESH_BINARY)
    
    # Região de interesse - metade inferior da imagem
    altura, largura = binario.shape
    mascara = np.zeros_like(binario)
    roi_vertices = np.array([[(0, altura), (0, altura//2), (largura, altura//2), (largura, altura)]], dtype=np.int32)
    cv2.fillPoly(mascara, roi_vertices, 255)
    binario_roi = cv2.bitwise_and(binario, mascara)
    
    # Criar imagem para visualização
    img_debug = np.dstack((binario_roi, binario_roi, binario_roi))
    
    # Calcular histograma da metade inferior da imagem
    histograma = np.sum(binario_roi[altura//2:, :], axis=0)
    
    # Encontrar picos no histograma para posições base das linhas
    metade_largura = largura//2
    esquerda_base = np.argmax(histograma[:metade_largura])
    direita_base = np.argmax(histograma[metade_largura:]) + metade_largura
    
    # Configurações das janelas
    n_janelas = 18
    altura_janela = altura//n_janelas
    margem = 100  # Largura da janela
    min_pixels = 50  # Mínimo de pixels para recentrar a janela
    
    # Identificar pixels não-zero na imagem
    pixels_nao_zero = binario_roi.nonzero()
    y_nao_zero = np.array(pixels_nao_zero[0])
    x_nao_zero = np.array(pixels_nao_zero[1])
    
    # Posições atuais que serão atualizadas para cada janela
    esquerda_atual = esquerda_base
    direita_atual = direita_base
    
    # Listas para guardar os índices
    indices_esquerda = []
    indices_direita = []
    
    # Percorrer janelas de baixo para cima
    for janela in range(n_janelas):
        # Limites verticais da janela
        y_baixo = altura - (janela+1)*altura_janela
        y_cima = altura - janela*altura_janela
        
        # Limites horizontais das janelas
        esq_x_esquerda = esquerda_atual - margem
        esq_x_direita = esquerda_atual + margem
        dir_x_esquerda = direita_atual - margem
        dir_x_direita = direita_atual + margem
        
        # Desenhar as janelas na imagem de debug
        cv2.rectangle(img_debug, (esq_x_esquerda, y_baixo), (esq_x_direita, y_cima), (0, 255, 0), 2) 
        cv2.rectangle(img_debug, (dir_x_esquerda, y_baixo), (dir_x_direita, y_cima), (0, 255, 0), 2) 
        
        # Identificar pixels não-zero dentro das janelas
        bons_esquerda = ((y_nao_zero >= y_baixo) & (y_nao_zero < y_cima) & 
                          (x_nao_zero >= esq_x_esquerda) & (x_nao_zero < esq_x_direita)).nonzero()[0]
        bons_direita = ((y_nao_zero >= y_baixo) & (y_nao_zero < y_cima) & 
                         (x_nao_zero >= dir_x_esquerda) & (x_nao_zero < dir_x_direita)).nonzero()[0]
        
        # Adicionar às listas
        indices_esquerda.append(bons_esquerda)
        indices_direita.append(bons_direita)
        
        # Recentrar janelas com base nos pixels encontrados
        if len(bons_esquerda) > min_pixels:
            esquerda_atual = int(np.mean(x_nao_zero[bons_esquerda]))
        if len(bons_direita) > min_pixels:        
            direita_atual = int(np.mean(x_nao_zero[bons_direita]))
    
    # Concatenar arrays de índices
    try:
        indices_esquerda = np.concatenate(indices_esquerda)
        indices_direita = np.concatenate(indices_direita)
    except ValueError:
        return imagem, img_debug, binario_roi, imagem.copy()
    
    # Extrair coordenadas dos pixels das linhas
    x_esquerda = x_nao_zero[indices_esquerda]
    y_esquerda = y_nao_zero[indices_esquerda] 
    x_direita = x_nao_zero[indices_direita]
    y_direita = y_nao_zero[indices_direita]
    
    # Ajustar polinómios de 2º grau às linhas (se houver pixels suficientes)
    imagem_linhas = np.zeros_like(imagem)
    
    try:
        if len(x_esquerda) > 0 and len(x_direita) > 0:
            # Ajustar polinómios
            coefs_esquerda = np.polyfit(y_esquerda, x_esquerda, 2)
            coefs_direita = np.polyfit(y_direita, x_direita, 2)
            
            # Gerar valores y para desenhar
            plot_y = np.linspace(0, altura-1, altura)
            
            # Calcular valores x correspondentes
            esquerda_x = coefs_esquerda[0]*plot_y**2 + coefs_esquerda[1]*plot_y + coefs_esquerda[2]
            direita_x = coefs_direita[0]*plot_y**2 + coefs_direita[1]*plot_y + coefs_direita[2]
            
            # Converter para inteiros
            esquerda_x = esquerda_x.astype(int)
            direita_x = direita_x.astype(int)
            plot_y = plot_y.astype(int)
            
            # Desenhar pontos e polinómios
            for i, y in enumerate(plot_y):
                if 0 <= y < altura:
                    if 0 <= esquerda_x[i] < largura:
                        img_debug[y, esquerda_x[i]] = [255, 0, 0]  # Vermelho para linha esquerda
                    if 0 <= direita_x[i] < largura:
                        img_debug[y, direita_x[i]] = [0, 0, 255]   # Azul para linha direita
            
            # Criar polígono entre as linhas
            pts_esquerda = np.array([np.transpose(np.vstack([esquerda_x, plot_y]))])
            pts_direita = np.array([np.flipud(np.transpose(np.vstack([direita_x, plot_y])))])
            pts = np.hstack((pts_esquerda, pts_direita))
            
            # Desenhar área da faixa
            cv2.fillPoly(imagem_linhas, np.int_([pts]), (0, 255, 0))
            
            # Marcar linhas individuais
            for i in range(len(plot_y)-1):
                if 0 <= plot_y[i] < altura and 0 <= plot_y[i+1] < altura:
                    if (0 <= esquerda_x[i] < largura and 0 <= esquerda_x[i+1] < largura):
                        cv2.line(imagem_linhas, (esquerda_x[i], plot_y[i]), 
                                 (esquerda_x[i+1], plot_y[i+1]), (255, 0, 0), 5)
                    if (0 <= direita_x[i] < largura and 0 <= direita_x[i+1] < largura):
                        cv2.line(imagem_linhas, (direita_x[i], plot_y[i]), 
                                 (direita_x[i+1], plot_y[i+1]), (0, 0, 255), 5)
    
    except Exception as e:
        print(f"Erro ao ajustar polinómios: {e}")

    resultado = cv2.addWeighted(imagem, 0.8, imagem_linhas, 1, 0)
    
    return resultado, img_debug, binario_roi, imagem_linhas

def detetar_linhas_estrada(imagem):
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
    ret, binario = cv2.threshold(imagem_roi, 200, 255, cv2.THRESH_BINARY)
    
    # Deteção de bordas
    bordas = cv2.Canny(binario, 50, 150)

                  
    cv2.imshow('BORDAS', bordas)
    
    # Criar imagem colorida para mostrar as bordas
    bordas_coloridas = cv2.cvtColor(bordas, cv2.COLOR_GRAY2BGR)
    
    # # Deteção simplificada de linhas (sem filtrar por direção/inclinação)
    linhas = cv2.HoughLinesP(
        bordas,
        rho=1,
        theta=np.pi/180,
        threshold=50,
        minLineLength=50,
        maxLineGap=100
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
                inclinacao = (y2 - y1) / (x2 - x1)
            
            # Filtrar linhas pela inclinação e posição
            if abs(inclinacao) < 0.5:  # Ignorar linhas muito horizontais
                continue
                
            if inclinacao < 0 and x1 < largura/2:
                linhas_esquerda.append(linha)
                cv2.line(imagem_linhas, (x1, y1), (x2, y2), (255, 0, 0), 2)  # Linha vermelha à esquerda
            elif inclinacao > 0 and x1 > largura/2:
                linhas_direita.append(linha)
                cv2.line(imagem_linhas, (x1, y1), (x2, y2), (0, 0, 255), 2)  # Linha azul à direita
    
    # Combinar as imagens
    resultado = cv2.addWeighted(imagem, 0.8, imagem_linhas, 1, 0)
    
    cv2.imshow('RESULTADO', resultado)
    
    # return resultado, imagem_linhas, binario, bordas_coloridas


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
        cv2.namedWindow('CARLA', cv2.WINDOW_NORMAL)
      
 
        cv2.resizeWindow('CARLA', 600, 300)
        
        # Criar sliders para os parâmetros
        # cv2.createTrackbar('Threshold Min', 'CARLA', threshold_min, 255, nothing)
        # cv2.createTrackbar('Threshold Max', 'CARLA', threshold_max, 255, nothing)
        # cv2.createTrackbar('Sobel Min', 'CARLA', sobel_min, 255, nothing)
        # cv2.createTrackbar('Sobel Max', 'CARLA', sobel_max, 255, nothing)
        # cv2.createTrackbar('Canny Min', 'CARLA', canny_min, 255, nothing)
        # cv2.createTrackbar('Canny Max', 'CARLA', canny_max, 255, nothing)
        
        # # Criar janela para visualização
        # cv2.namedWindow('DEBUG', cv2.WINDOW_NORMAL)
        # cv2.resizeWindow('DEBUG', LARGURA*2, ALTURA*2)
        
 
        # Função para processar a imagem da câmara
        def processar_imagem(imagem):
            nonlocal imagem_atual 

            # array = np.frombuffer(imagem.raw_data, dtype=np.dtype("uint8"))
            # array = np.reshape(array, (imagem.height, imagem.width, 4))
            # array = array[:, :, :3]  # Descarta o canal alfa
            # array = array[:, :, ::-1]  # Converte BGR para RGB
            # #imagem_opencv = array.copy()
            
            # Converter para formato OpenCV
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
                
                # Adicionar informações à imagem original
                velocidade = veiculo.get_velocity()
                rapidez = 3.6 * np.sqrt(velocidade.x**2 + velocidade.y**2 + velocidade.z**2)  # km/h
                info = f"Vel: {rapidez:.1f} km/h | Steer: {steering_atual:.2f}"
                cv2.putText(img_original, info, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.putText(img_original, "ESC: Sair | W/S: Acel/Trav | A/D: Virar", (10, ALTURA - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                
                
               
                
                cv2.imshow('CARLA', img_original)

                #detetar_linhas_estrada(imagem_atual)
                # imagem_bgr = cv2.cvtColor(imagem_atual, cv2.COLOR_RGB2BGR)
    
 

                resultado, img_debug, binario, linhas = detetar_linhas_janela_deslizante(img_original)
                
                # Redimensionar imagens para visualização lado a lado
                altura, largura = img_original.shape[:2]
                binario_rgb = cv2.cvtColor(binario, cv2.COLOR_GRAY2BGR)
                
                # Criar grelha para visualização
                topo = np.hstack((img_original, img_debug))
                baixo = np.hstack((binario_rgb, linhas))
                grelha = np.vstack((topo, baixo))
                
                # Redimensionar para caber no ecrã
                escala = min(1.0, 1280 / grelha.shape[1])
                dim = (int(grelha.shape[1] * escala), int(grelha.shape[0] * escala))
                grelha_redim = cv2.resize(grelha, dim)
                
                # Mostrar a grelha
                cv2.imshow("Lines", grelha_redim)
                
                #final = cv2.cvtColor(resultado, cv2.COLOR_BGR2RGB)
                #line_detection(imagem_atual)

                
        
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