import pygame
import numpy as np
from essential_functions import Univerzalna_Advekcija, Univerzalna_Difuzija, IndexMap, vectorB, Matrica_A, IzracunajPritisak

def ubrizgaj_brzinu(brzina_x, brzina_y, centar_i, centar_j, radijus, v_x, v_y):
    
    for i in range(max(0, centar_i - radijus), min(brzina_x.shape[0], centar_i + radijus + 1)):
        for j in range(max(0, centar_j - radijus), min(brzina_x.shape[1], centar_j + radijus + 1)):
            if (i - centar_i)**2 + (j - centar_j)**2 <= radijus**2:
                brzina_x[i, j] = v_x  
                
    for i in range(max(0, centar_i - radijus), min(brzina_y.shape[0], centar_i + radijus + 1)):
        for j in range(max(0, centar_j - radijus), min(brzina_y.shape[1], centar_j + radijus + 1)):
            if (i - centar_i)**2 + (j - centar_j)**2 <= radijus**2:
                brzina_y[i, j] = v_y

# ==========================================================
# KONFIGURACIJA
# ==========================================================
N = 32
dt = 0.01 
h = 0.1
ni = 0.1    
rho = 1.0
REZOLUCIJA = 720
skala = REZOLUCIJA // N

tip_celije = np.ones((N, N), dtype=int)
tip_celije[0, :] = 0; tip_celije[-1, :] = 0; tip_celije[:, 0] = 0; tip_celije[:, -1] = 0

mapa_indexa = IndexMap(tip_celije)
A_sistem = Matrica_A(tip_celije)

brzina_x = np.ones((N, N+1))
brzina_y = np.zeros((N+1, N))

# Početni mehur brzine
centar_i, centar_j = N // 2, N // 4
radijus = 4
for i in range(N):
    for j in range(N+1):
        if (i - centar_i)**2 + (j - centar_j)**2 < radijus**2: brzina_x[i, j] = 10.0
for i in range(N+1):
    for j in range(N):
        if (i - centar_i)**2 + (j - centar_j)**2 < radijus**2: brzina_y[i, j] = 10.0

# ==========================================================
# PYGAME SETUP
# ==========================================================
pygame.init()
screen = pygame.display.set_mode((REZOLUCIJA, REZOLUCIJA))
clock = pygame.time.Clock()

def get_color(val):
    
    val = np.clip(val, -5, 5)
    norm = (val + 5) / 10
    return (int(255 * norm), 0, int(255 * (1 - norm)))

# ==========================================================
# GLAVNA PETLJA
# ==========================================================
running = True
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT: running = False

    
    b_vektor = vectorB(tip_celije, brzina_x, brzina_y, rho, dt, h)
    P_matrica = IzracunajPritisak(A_sistem, b_vektor, mapa_indexa, tip_celije, tol=1e-5)
    P_matrica *= 0.98  



    for i in range(N):
        for j in range(1, N):
            if tip_celije[i, j] != 0 and tip_celije[i, j-1] != 0:
                brzina_x[i, j] -= (dt / (rho * h)) * (P_matrica[i, j] - P_matrica[i, j-1])
            else: brzina_x[i, j] = 0.0
            
    for i in range(1, N):
        for j in range(N):
            if tip_celije[i, j] != 0 and tip_celije[i-1, j] != 0:
                brzina_y[i, j] -= (dt / (rho * h)) * (P_matrica[i, j] - P_matrica[i-1, j])
            else: brzina_y[i, j] = 0.0

    brzina_x = Univerzalna_Difuzija(Univerzalna_Advekcija(brzina_x, brzina_x, brzina_y, 'x_ivica', dt, h), ni, dt, h)
    brzina_y = Univerzalna_Difuzija(Univerzalna_Advekcija(brzina_y, brzina_x, brzina_y, 'y_ivica', dt, h), ni, dt, h)
    
    screen.fill((0, 0, 0))
    for i in range(N):
        for j in range(N):
            
            pygame.draw.rect(screen, get_color(P_matrica[i, j]), (j*skala, i*skala, skala, skala))
            

    pygame.display.flip()
    clock.tick(30)

pygame.quit()