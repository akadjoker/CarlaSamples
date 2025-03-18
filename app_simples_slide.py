import carla
import cv2
import numpy as np
import time
import random
import sys

 

# Configurações
LARGURA = 640
ALTURA = 480

 
# Valores iniciais para os sliders
thresh_value = 200
n_janelas = 18
margem = 100
min_pixels = 50

# Valores iniciais para o ROI
roi_top_left_x = 0
roi_top_left_y = ALTURA//2
roi_top_right_x = LARGURA
roi_top_right_y = ALTURA//2
roi_bottom_right_x = LARGURA
roi_bottom_right_y = ALTURA
roi_bottom_left_x = 0
roi_bottom_left_y = ALTURA

def nothing(x):
    pass

def detetar_linhas_janela_deslizante(imagem, thresh_val, n_windows, margin, minpix, roi_points, prev_left=None, prev_right=None):
    """
    Versão melhorada com estabilização dos ângulos
    """
    # Converter para escala de cinzentos
    cinza = cv2.cvtColor(imagem, cv2.COLOR_BGR2GRAY)
    
    # Aplicar limiar para isolar linhas brancas
    _, binario = cv2.threshold(cinza, thresh_val, 255, cv2.THRESH_BINARY)
    
    # Região de interesse usando os pontos configuráveis
    altura, largura = binario.shape
    mascara = np.zeros_like(binario)
    roi_vertices = np.array([roi_points], dtype=np.int32)
    cv2.fillPoly(mascara, roi_vertices, 255)
    binario_roi = cv2.bitwise_and(binario, mascara)
    
    # Criar imagem para visualização
    img_debug = np.dstack((binario_roi, binario_roi, binario_roi))
    
    # Desenhar ROI na imagem de debug
    cv2.polylines(img_debug, [np.array(roi_points, dtype=np.int32)], True, (0, 255, 255), 2)
    
    # Calcular histograma da metade inferior da imagem
    histograma = np.sum(binario_roi[altura//2:, :], axis=0)
    
    # Encontrar picos no histograma para posições base das linhas
    metade_largura = largura//2
    
    # Se tivermos deteções anteriores válidas, usá-las como referência
    if prev_left is not None and prev_right is not None:
        # Usar uma janela em torno da posição anterior
        janela_esq = 100  # Ajustar conforme necessário
        janela_dir = 100
        
        # Limitar a pesquisa em torno da posição anterior
        esq_min = max(0, prev_left - janela_esq)
        esq_max = min(metade_largura, prev_left + janela_esq)
        dir_min = max(metade_largura, prev_right - janela_dir)
        dir_max = min(largura, prev_right + janela_dir)
        
        # Encontrar picos dentro das janelas
        esquerda_base = esq_min + np.argmax(histograma[esq_min:esq_max])
        direita_base = dir_min + np.argmax(histograma[dir_min:dir_max])
    else:
        # Primeira deteção ou deteção falhou no frame anterior
        esquerda_base = np.argmax(histograma[:metade_largura])
        direita_base = np.argmax(histograma[metade_largura:]) + metade_largura
    
    # Configurações das janelas
    altura_janela = altura//n_windows
    
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
    
    # Listas para guardar as posições das janelas
    posicoes_esquerda = []
    posicoes_direita = []
    
    # Percorrer janelas de baixo para cima
    for janela in range(n_windows):
        # Limites verticais da janela
        y_baixo = altura - (janela+1)*altura_janela
        y_cima = altura - janela*altura_janela
        
        # Limites horizontais das janelas
        esq_x_esquerda = esquerda_atual - margin
        esq_x_direita = esquerda_atual + margin
        dir_x_esquerda = direita_atual - margin
        dir_x_direita = direita_atual + margin
        
        # Impedir que as janelas se sobreponham ou cruzem
        if esq_x_direita > dir_x_esquerda:
            # Ajustar para manter separação mínima
            meio = (esq_x_direita + dir_x_esquerda) // 2
            esq_x_direita = meio - 10
            dir_x_esquerda = meio + 10
        
        # Guardar as posições das janelas
        posicoes_esquerda.append((esquerda_atual, (y_baixo + y_cima) // 2))
        posicoes_direita.append((direita_atual, (y_baixo + y_cima) // 2))
        
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
        
        # Guardar posição anterior para limitar movimento
        esquerda_anterior = esquerda_atual
        direita_anterior = direita_atual
        
        # Recentrar janelas com base nos pixels encontrados
        if len(bons_esquerda) > minpix:
            novo_esquerda = int(np.mean(x_nao_zero[bons_esquerda]))
            # Limitar movimento da janela
            max_movimento = margin // 2
            if abs(novo_esquerda - esquerda_anterior) < max_movimento:
                esquerda_atual = novo_esquerda
        
        if len(bons_direita) > minpix:
            novo_direita = int(np.mean(x_nao_zero[bons_direita]))
            # Limitar movimento da janela
            max_movimento = margin // 2
            if abs(novo_direita - direita_anterior) < max_movimento:
                direita_atual = novo_direita
        
        # Garantir que as janelas não cruzam o meio da imagem
        if esquerda_atual > metade_largura - 20:
            esquerda_atual = metade_largura - 20
        if direita_atual < metade_largura + 20:
            direita_atual = metade_largura + 20
    
    # Concatenar arrays de índices
    try:
        indices_esquerda = np.concatenate(indices_esquerda)
        indices_direita = np.concatenate(indices_direita)
    except ValueError:
        # Se não houver pixels suficientes, retornar apenas a imagem original
        return imagem, img_debug, binario_roi, imagem.copy()
    
    # Extrair coordenadas dos pixels das linhas
    x_esquerda = x_nao_zero[indices_esquerda]
    y_esquerda = y_nao_zero[indices_esquerda] 
    x_direita = x_nao_zero[indices_direita]
    y_direita = y_nao_zero[indices_direita]
    
    # Ajustar polinómios de 2º grau às linhas (se houver pixels suficientes)
    imagem_linhas = np.zeros_like(imagem)
    
    try:
        if len(x_esquerda) > 10 and len(x_direita) > 10:  # Mínimo de pontos para ajustar
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
        return imagem, img_debug, binario_roi, imagem.copy()

    resultado = cv2.addWeighted(imagem, 0.8, imagem_linhas, 1, 0)
    
    return resultado, img_debug, binario_roi, imagem_linhas, esquerda_base, direita_base

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
    
    cv2.imshow('CARLA', resultado)
 


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
        cv2.resizeWindow('Controlos', 600, 400)
        
        # Criar sliders para os parâmetros
        cv2.createTrackbar('Threshold', 'Controlos', thresh_value, 255, nothing)
        cv2.createTrackbar('Num Janelas', 'Controlos', n_janelas, 50, nothing)
        cv2.createTrackbar('Margem', 'Controlos', margem, 200, nothing)
        cv2.createTrackbar('Min Pixels', 'Controlos', min_pixels, 200, nothing)
        
        # Sliders para o ROI
        cv2.createTrackbar('ROI Topo X', 'Controlos', LARGURA//4, LARGURA, nothing)
        cv2.createTrackbar('ROI Topo Y', 'Controlos', ALTURA//2, ALTURA, nothing)
        cv2.createTrackbar('ROI Larg', 'Controlos', LARGURA//2, LARGURA, nothing)
        
        # Criar janela para visualização
        cv2.namedWindow('Debug', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('Debug', LARGURA*2, ALTURA*2)
        
 
   
        
 
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
                img_original =imagem_atual.copy()# cv2.cvtColor(imagem_atual, cv2.COLOR_RGB2BGR)
                #img_original = imagem_atual
                
                # Adicionar informações à imagem original
                velocidade = veiculo.get_velocity()
                rapidez = 3.6 * np.sqrt(velocidade.x**2 + velocidade.y**2 + velocidade.z**2)  # km/h
                info = f"Vel: {rapidez:.1f} km/h | Steer: {steering_atual:.2f}"
                cv2.putText(img_original, info, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.putText(img_original, "ESC: Sair | W/S: Acel/Trav | A/D: Virar", (10, ALTURA - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                
                
               
                detetar_linhas_estrada(img_original)
                cv2.imshow('CARLA', img_original)

                #detetar_linhas_estrada(imagem_atual)
                # imagem_bgr = cv2.cvtColor(imagem_atual, cv2.COLOR_RGB2BGR)
    
 
                imagem_bgr = cv2.cvtColor(imagem_atual, cv2.COLOR_RGB2BGR)
                 # Ler valores dos sliders
                thresh_val = cv2.getTrackbarPos('Threshold', 'Controlos')
                n_windows = cv2.getTrackbarPos('Num Janelas', 'Controlos')
                if n_windows < 1: n_windows = 1  # Evitar divisão por zero
                margin = cv2.getTrackbarPos('Margem', 'Controlos')
                minpix = cv2.getTrackbarPos('Min Pixels', 'Controlos')
                
                # Ler valores do ROI
                roi_topo_x = cv2.getTrackbarPos('ROI Topo X', 'Controlos')
                roi_topo_y = cv2.getTrackbarPos('ROI Topo Y', 'Controlos')
                roi_largura = cv2.getTrackbarPos('ROI Larg', 'Controlos')
                
                # Calcular os pontos do ROI baseados nos sliders
                roi_points = [
                    [roi_topo_x, roi_topo_y],                      # Topo esquerdo
                    [roi_topo_x + roi_largura, roi_topo_y],        # Topo direito
                    [LARGURA, ALTURA],                             # Baixo direito
                    [0, ALTURA]                                    # Baixo esquerdo
                ]
                
                resultado, debug, binario, linhas, left_base, right_base = detetar_linhas_janela_deslizante(
                    imagem_bgr, thresh_val, n_windows, margin, minpix, roi_points, left_base_prev, right_base_prev
                )
                
       
                left_base_prev = left_base
                right_base_prev = right_base
                    
                    # Mostrar o resultado
                topo = np.hstack((imagem_bgr, resultado))
                baixo = np.hstack((debug, cv2.cvtColor(binario, cv2.COLOR_GRAY2BGR)))
                
                # Redimensionar se necessário para igualar dimensões
                if baixo.shape[1] != topo.shape[1]:
                    baixo = cv2.resize(baixo, (topo.shape[1], baixo.shape[0]))
                
                # Juntar visualizações
                visualizacao = np.vstack((topo, baixo))
                
                # Adicionar informações
                info_text = f"Thresh: {thresh_val} | Janelas: {n_windows} | Margem: {margin} | Min Pixels: {minpix}"
                cv2.putText(visualizacao, info_text, (10, 30), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                
                # Mostrar a visualização
                cv2.imshow('Debug', visualizacao)
        
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