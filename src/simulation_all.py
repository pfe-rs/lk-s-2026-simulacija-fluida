import pygame
import numpy as np
# Uvozimo tvoje funkcije iz projekta
from essential_functions import IndexMap, Matrica_A, vectorB, IzracunajPritisak, Univerzalna_Advekcija, Univerzalna_Difuzija

# --- 1. FUNKCIJE ZA PAMETAN RAD SA SILAMA I BOJAMA ---

def dodaj_impuls_sile(sila_x, sila_y, centar_i, centar_j, radijus, jacina_x, jacina_y):
    """
    Dodaje silu u obliku kruga oko pozicije miša na Staggered Grid.
    """
    N_redova_x, N_kolona_x = sila_x.shape
    N_redova_y, N_kolona_y = sila_y.shape
    
    # Horizontalna sila (sila_x)
    for i in range(N_redova_x):
        for j in range(N_kolona_x):
            if (i - centar_i)**2 + (j - centar_j)**2 < radijus**2:
                sila_x[i, j] += jacina_x
                
    # Vertikalna sila (sila_y)
    for i in range(N_redova_y):
        for j in range(N_kolona_y):
            if (i - centar_i)**2 + (j - centar_j)**2 < radijus**2:
                sila_y[i, j] += jacina_y

def odredi_boju(vrednost, max_vrednost=0.5):
    """
    Pretvara pritisak u RGB boju (Plava za nizak, Zelena za nulu, Crvena za visok).
    Povećao sam max_vrednost na 0.5 jer miš pravi nagle skokove pritiska!
    """
    izbalansirano = np.clip(vrednost / max_vrednost, -1.0, 1.0)
    
    if izbalansirano < 0:
        faktor = abs(izbalansirano)  
        r = 0
        g = int(255 * (1.0 - faktor))
        b = int(255 * faktor)
    else:
        faktor = izbalansirano   
        r = int(255 * faktor)
        g = int(255 * (1.0 - faktor))
        b = 0
        
    return (r, g, b)

# --- 2. KONSTANTE I SIMULACIONE MATRICE ---
N = 32
dt = 0.05
h = 0.1
rho = 1.0
ni = 0.1 # Koeficijent viskoznosti

# Inicijalizacija brzina i sila
brzina_x = np.zeros((N, N + 1))
brzina_y = np.zeros((N + 1, N))
sila_x = np.zeros((N, N + 1))
sila_y = np.zeros((N + 1, N))

# Kreiramo tip ćelije (Zidovi na ivicama, unutra fluid)
tip_celije = np.ones((N, N))
tip_celije[0, :] = 0   # Gornji zid
tip_celije[-1, :] = 0  # Donji zid
tip_celije[:, 0] = 0   # Levi zid
tip_celije[:, -1] = 0  # Desni zid

# Unapred računamo mapu indeksa i statičku matricu sistema A
mapa_indexa = IndexMap(tip_celije)
A_sistem = Matrica_A(tip_celije)

# --- 3. PYGAME INICIJALIZACIJA ---
pygame.init()
REZOLUCIJA = 512
prozor = pygame.display.set_mode((REZOLUCIJA, REZOLUCIJA))
pygame.display.set_caption("Interaktivna Fluid Simulacija (PyGame)")
sat = pygame.time.Clock() # Za kontrolu FPS-a

skala = REZOLUCIJA // N

# Glavna petlja aplikacije
running = True
while running:
    # Resetujemo akumulirane eksterne sile na nulu za ovaj frejm
    sila_x[:, :] = 0.0
    sila_y[:, :] = 0.0

    # 1. Provera sistemskih događaja
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    # 2. INTERAKTIVNO GURANJE MIŠEM
    mis_pritisnut = pygame.mouse.get_pressed()
    if mis_pritisnut[0]:  # Levi klik drži guranje
        pos_x, pos_y = pygame.mouse.get_pos()
        
        # Preslikavanje piksela u indekse matrice (0 do N-1)
        mis_j = pos_x // skala
        mis_i = pos_y // skala
        
        # Uzimamo brzinu pomeranja miša
        brzina_misa_x, brzina_misa_y = pygame.mouse.get_rel()
        
        # Dodajemo impuls u fluid (pomnoženo sa 20.0 da osetiš otpor)
        if 0 < mis_i < N-1 and 0 < mis_j < N-1:
            dodaj_impuls_sile(sila_x, sila_y, 
                              centar_i=mis_i, centar_j=mis_j, radijus=2, 
                              jacina_x=brzina_misa_x * 20.0, 
                              jacina_y=brzina_misa_y * 20.0)
    else:
        # Re-inicijalizacija relativnog pomaka miša kada klik nije pritisnut,
        # da ne bi dobili ogroman skok pri sledećem kliku
        pygame.mouse.get_rel()

    # --- 3. FIZIČKI PALEONTOLOŠKI PIPELINE ---
    
    # Korak 0: Primena eksternih sila na trenutno polje brzina
    brzina_x += sila_x * dt
    brzina_y += sila_y * dt
    
    # Korak 1 & 2: Rešavanje pritiska (Projektovanje)
    b_vektor = vectorB(tip_celije, brzina_x, brzina_y, rho, dt, h)
    P_matrica = IzracunajPritisak(A_sistem, b_vektor, mapa_indexa, tip_celije, tol=1e-4)

    # Korak 3: Korekcija brzina pomoću dobijenog pritiska + očuvanje tvrdih zidova
    for i in range(N):
        for j in range(1, N):
            if tip_celije[i, j] != 0 and tip_celije[i, j-1] != 0:
                brzina_x[i, j] = brzina_x[i, j] - (dt / (rho * h)) * (P_matrica[i, j] - P_matrica[i, j-1])
            else:
                brzina_x[i, j] = 0.0
            
    for i in range(1, N):
        for j in range(N):
            if tip_celije[i, j] != 0 and tip_celije[i-1, j] != 0:
                brzina_y[i, j] = brzina_y[i, j] - (dt / (rho * h)) * (P_matrica[i, j] - P_matrica[i-1, j])
            else:
                brzina_y[i, j] = 0.0

    # Korak 4: Advekcija (Premeštanje fluida niz sopstvenu brzinu)
    brzina_x = Univerzalna_Advekcija(brzina_x, brzina_x, brzina_y, 'x_ivica', dt, h)
    brzina_y = Univerzalna_Advekcija(brzina_y, brzina_x, brzina_y, 'y_ivica', dt, h)

    # Korak 5: Difuzija (Viskoznost) + Damping (Prigušenje) za stabilizaciju
    brzina_x = Univerzalna_Difuzija(brzina_x, ni, dt, h) * 0.995
    brzina_y = Univerzalna_Difuzija(brzina_y, ni, dt, h) * 0.995

    # --- 4. RENDER ELEMENTA NA EKRAN ---
    prozor.fill((0, 0, 0)) # Čistimo ekran
    
    for i in range(N):
        for j in range(N):
            if tip_celije[i, j] == 0:
                # Elegantna tamna boja za zidove kutije
                boja = (15, 23, 42)
            else:
                # Renderujemo pritisak u plavo-beloj skali
                boja = odredi_boju(P_matrica[i, j], max_vrednost=0.15) # Blago spušten max za bolji kontrast
                
            pygame.draw.rect(prozor, boja, (j * skala, i * skala, skala, skala))

    pygame.display.flip()
    sat.tick(60)

pygame.quit()