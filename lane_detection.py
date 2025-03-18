import numpy as np
import cv2

# Classe para rastrear as linhas de faixa
class Line():
    def __init__(self):
        # Foi a linha detectada no último quadro?
        self.detected = False
        
        # Coeficientes polinomiais para o ajuste mais recente
        self.current_fit = [np.array([False])]
        
        # Coeficientes polinomiais para os últimos n ajustes
        self.recent_fits = []
        
        # Número máximo de ajustes recentes a armazenar
        self.max_recent = 5
        
        # Valores x dos pixels da linha no último quadro
        self.current_x = None
        
        # Ajuste polinomial médio nos últimos n quadros
        self.best_fit = None
        
        # Valores x correspondentes ao ajuste polinomial médio
        self.best_x = None
        
        # Raio de curvatura da linha em metros
        self.radius_of_curvature = None
        
        # Distância em metros do centro do veículo até o lado da faixa
        self.line_base_pos = None
        
    def update_fit(self, fit):
        # Adiciona um novo ajuste polinomial à lista de ajustes recentes
        self.current_fit = fit
        self.recent_fits.append(fit)
        
        # Mantém apenas o número máximo de ajustes recentes
        if len(self.recent_fits) > self.max_recent:
            self.recent_fits.pop(0)
        
        # Calcula o ajuste médio a partir dos ajustes recentes
        self.best_fit = np.mean(self.recent_fits, axis=0)
        
        # Marca a linha como detectada
        self.detected = True
    
    def calculate_x(self, ploty):
        # Calcula os valores x para os valores y dados usando o melhor ajuste
        if self.best_fit is not None:
            self.best_x = self.best_fit[0]*ploty**2 + self.best_fit[1]*ploty + self.best_fit[2]
        return self.best_x
    
    def calculate_curvature(self, ploty):
        # Calculando o raio de curvatura em metros
        # Define conversões em x e y de pixels para metros
        ym_per_pix = 30/720 # metros por pixel na direção y
        xm_per_pix = 3.7/700 # metros por pixel na direção x
        
        # Ponto no qual queremos o raio de curvatura (parte inferior da imagem)
        y_eval = np.max(ploty)
        
        # Ajustar um novo polinômio em espaço métrico
        fit_cr = np.polyfit(ploty*ym_per_pix, self.best_x*xm_per_pix, 2)
        
        # Calcular o raio de curvatura
        self.radius_of_curvature = ((1 + (2*fit_cr[0]*y_eval*ym_per_pix + fit_cr[1])**2)**1.5) / np.absolute(2*fit_cr[0])
        
        return self.radius_of_curvature
    
    def calculate_line_base_pos(self, img_width):
        # Calcula a posição da linha na base da imagem em relação ao centro
        # Assumindo que a câmera está montada no centro do veículo
        center_px = img_width // 2
        
        # Converter para metros
        xm_per_pix = 3.7/700 # metros por pixel na direção x
        
        if self.best_x is not None:
            # Posição da linha na base da imagem (parte inferior)
            line_base_px = self.best_x[-1]
            
            # Calcular a distância do centro do veículo
            self.line_base_pos = (line_base_px - center_px) * xm_per_pix
        
        return self.line_base_pos

# Criar objetos globais para rastrear as linhas da esquerda e direita
left_line = Line()
right_line = Line()

def find_lane_pixels(binary_warped):
    """
    Usa o método do histograma e janelas deslizantes para encontrar pixels das faixas
    """
    # Criar uma imagem de saída para desenhar e visualizar
    out_img = np.dstack((binary_warped, binary_warped, binary_warped)) * 255
    
    # Calcular histograma da metade inferior da imagem
    histogram = np.sum(binary_warped[binary_warped.shape[0]//2:,:], axis=0)
    
    # Encontrar os picos do histograma para as metades esquerda e direita
    midpoint = np.int_(histogram.shape[0]//2)
    leftx_base = np.argmax(histogram[:midpoint])
    rightx_base = np.argmax(histogram[midpoint:]) + midpoint

    # Parâmetros para as janelas deslizantes
    nwindows = 9  # Número de janelas
    margin = 100  # Largura da janela
    minpix = 50   # Mínimo de pixels para recentrar a janela

    # Altura de cada janela
    window_height = np.int_(binary_warped.shape[0]//nwindows)
    
    # Identificar as posições (x,y) de todos os pixels não-zero
    nonzero = binary_warped.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    
    # Posições atuais a serem atualizadas para cada janela
    leftx_current = leftx_base
    rightx_current = rightx_base
    
    # Listas vazias para receber os índices dos pixels da esquerda e direita
    left_lane_inds = []
    right_lane_inds = []

    # Percorrer as janelas uma a uma
    for window in range(nwindows):
        # Identificar os limites da janela em x e y
        win_y_low = binary_warped.shape[0] - (window+1)*window_height
        win_y_high = binary_warped.shape[0] - window*window_height
        
        win_xleft_low = leftx_current - margin
        win_xleft_high = leftx_current + margin
        win_xright_low = rightx_current - margin
        win_xright_high = rightx_current + margin
        
        # Desenhar as janelas na imagem de visualização
        cv2.rectangle(out_img, (win_xleft_low, win_y_low), (win_xleft_high, win_y_high), (0,255,0), 2) 
        cv2.rectangle(out_img, (win_xright_low, win_y_low), (win_xright_high, win_y_high), (0,255,0), 2) 
        
        # Identificar os pixels não-zero na janela
        good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & 
                          (nonzerox >= win_xleft_low) & (nonzerox < win_xleft_high)).nonzero()[0]
        good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & 
                           (nonzerox >= win_xright_low) & (nonzerox < win_xright_high)).nonzero()[0]
        
        # Adicionar às listas
        left_lane_inds.append(good_left_inds)
        right_lane_inds.append(good_right_inds)
        
        # Se encontrou pixels suficientes, recentrar a próxima janela
        if len(good_left_inds) > minpix:
            leftx_current = np.int_(np.mean(nonzerox[good_left_inds]))
        if len(good_right_inds) > minpix:        
            rightx_current = np.int_(np.mean(nonzerox[good_right_inds]))

    # Concatenar os arrays de índices
    try:
        left_lane_inds = np.concatenate(left_lane_inds)
        right_lane_inds = np.concatenate(right_lane_inds)
    except ValueError:
        # Evitar erro se não houver pixels suficientes
        pass
    
    # Extrair as posições dos pixels das linhas
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds] 
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds]
    
    return leftx, lefty, rightx, righty, out_img

def fit_polynomial(binary_warped):
    """
    Ajustar um polinômio de segundo grau às linhas de faixa
    """
    global left_line, right_line
    
    # Encontrar os pixels das faixas
    leftx, lefty, rightx, righty, out_img = find_lane_pixels(binary_warped)
    
    left_fit = None
    right_fit = None
    
    # Verificar se há pixels suficientes para ajustar
    if len(leftx) > 500:
        # Ajustar um polinômio de segundo grau
        left_fit = np.polyfit(lefty, leftx, 2)
        left_line.update_fit(left_fit)
    elif left_line.detected:
        # Usar o ajuste anterior
        left_fit = left_line.best_fit
    else:
        left_line.detected = False
        
    if len(rightx) > 500:
        # Ajustar um polinômio de segundo grau
        right_fit = np.polyfit(righty, rightx, 2)
        right_line.update_fit(right_fit)
    elif right_line.detected:
        # Usar o ajuste anterior
        right_fit = right_line.best_fit
    else:
        right_line.detected = False
    
    # Gerar valores x e y para plotar
    ploty = np.linspace(0, binary_warped.shape[0]-1, binary_warped.shape[0])
    
    try:
        # Calcular valores x para cada linha
        left_fitx = left_line.calculate_x(ploty)
        right_fitx = right_line.calculate_x(ploty)
        
        # Verificações de sanidade
        lane_width = np.mean(right_fitx - left_fitx)
        
        # Verificar se a largura da faixa é razoável
        if not (600 < lane_width < 900):
            # Resetar a detecção se a largura não for razoável
            left_line.detected = False
            right_line.detected = False
            return None, None, None, None, ploty
    except:
        # Em caso de erro, resetar a detecção
        left_line.detected = False
        right_line.detected = False
        return None, None, None, None, ploty
    
    # Colorir os pixels das faixas
    out_img[lefty, leftx] = [255, 0, 0]
    out_img[righty, rightx] = [0, 0, 255]
    
    # Criar resultado visual
    result_img = np.copy(out_img)
    
    # Desenhar o polígono da faixa
    if left_fitx is not None and right_fitx is not None:
        # Criar pontos para um polígono
        pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
        pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
        pts = np.hstack((pts_left, pts_right))
        
        # Desenhar o polígono na imagem
        cv2.fillPoly(result_img, np.int_([pts]), (0, 255, 0))
    
    # Calcular raio de curvatura
    left_curverad = left_line.calculate_curvature(ploty)
    right_curverad = right_line.calculate_curvature(ploty)
    
    # Calcular posição do veículo
    left_line.calculate_line_base_pos(binary_warped.shape[1])
    right_line.calculate_line_base_pos(binary_warped.shape[1])
    
    return left_fitx, right_fitx, result_img, out_img, ploty
