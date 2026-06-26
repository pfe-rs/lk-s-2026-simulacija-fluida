import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import lil_matrix
from scipy.sparse.linalg import cg

# Dimenzije mrežice (N x N ćelija)
N = 16
h = 0.1
dt = 0.01
rho = 1.0

# 1. Inicijalizacija polja brzina
brzina_x = np.zeros((N, N+1))
brzina_y = np.zeros((N+1, N))

# Mapiranje ćelija: 1 = FLUID, 0 = SOLID (Zidovi su na ivicama kutije)
Tip_Celije = np.zeros((N, N), dtype=int)
Tip_Celije[1:-1, 1:-1] = 1 # Unutrašnjost je fluid, ivice mrežice su zidovi

# SCENARIO: Dve struje udaraju jedna u drugu u centru (i=8)
brzina_x[8, 2:8] = 10.0   # Struja sleva ide udesno
brzina_x[8, 9:15] = -10.0 # Struja zdesna ide ulevo

# indeksiranje fluidnih ćelija za matricu A
fluidne_celije = []
mapa_indeksa = -np.ones((N, N), dtype=int)
brojac = 0

for i in range(N):
    for j in range(N):
        if Tip_Celije[i, j] == 1: # Ako je fluid
            fluidne_celije.append((i, j))
            mapa_indeksa[i, j] = brojac
            brojac += 1

M = len(fluidne_celije) # Broj nepoznatih pritisaka

# 2. FORMIRANJE MATRICE A I VEKTORA B
A = lil_matrix((M, M))
B = np.zeros(M)

for idx, (i, j) in enumerate(fluidne_celije):
    ne_solidni_komšije = 0
    
    # Provera 4 komšije: (gore, dole, levo, desno)
    komšije = [(i-1, j), (i+1, j), (i, j-1), (i, j+1)]
    
    for ki, kj in komšije:
        # Ako komšija nije zid (unutar mrežice je i tip je FLUID)
        # (U našoj jednostavnoj kutiji, sve što nije fluid je solid/zid)
        if 0 <= ki < N and 0 <= kj < N and Tip_Celije[ki, kj] == 1:
            ne_solidni_komšije += 1
            k_idx = mapa_indeksa[ki, kj]
            A[idx, k_idx] = 1.0
        elif 0 <= ki < N and 0 <= kj < N:
            # Komšija je unutar mrežice ali je solid
            pass
        else:
            # Izvan mrežice se smatra solid zidom
            pass
            
    # Glavna dijagonala: minus broj ne-solidnih komšija
    # Pošto nemamo vazduh (AIR) u ovoj kutiji, ne_solidni_komšije su zapravo fluidni komšije
    A[idx, idx] = -ne_solidni_komšije
    
    # Računanje modifikovane divergencije za vektor B
    # Uzimamo u obzir ivice ćelije (i, j)
    div = (brzina_x[i, j+1] - brzina_x[i, j]) / h + (brzina_y[i+1, j] - brzina_y[i, j]) / h
    B[idx] = (rho * h / dt) * div

# Prebacujemo A u CSR format radi ekstremno brzog rešavanja
A = A.tocsr()

# 3. REŠAVANJE SISTEMA PREKO CONJUGATE GRADIENT (Iz SciPy-a)
P_vektor, info = cg(A, B, tol=1e-5)

# Smeštamo pritiske nazad u 2D matricu
P_matrica = np.zeros((N, N))
for idx, (i, j) in enumerate(fluidne_celije):
    P_matrica[i, j] = P_vektor[idx]

# 4. PRIMENA PRITISKA NA BRZINE (Korekcija)
brzina_x_nova = np.copy(brzina_x)
brzina_y_nova = np.copy(brzina_y)

for i in range(N):
    for j in range(1, N):
        # Ažuriramo unutrašnje ivice X brzine ako je bar jedna ćelija fluid
        if Tip_Celije[i, j-1] == 1 or Tip_Celije[i, j] == 1:
            grad_p = (P_matrica[i, j] - P_matrica[i, j-1]) / h
            brzina_x_nova[i, j] = brzina_x[i, j] - (dt / (rho * h)) * (P_matrica[i, j] - P_matrica[i, j-1])

for i in range(1, N):
    for j in range(N):
        # Ažuriramo unutrašnje ivice Y brzine
        if Tip_Celije[i-1, j] == 1 or Tip_Celije[i, j] == 1:
            brzina_y_nova[i, j] = brzina_y[i, j] - (dt / (rho * h)) * (P_matrica[i, j] - P_matrica[i-1, j])

# --- VIZUELIZACIJA REZULTATA (Pre i Posle Pritiska) ---
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
X, Y = np.meshgrid(np.arange(N), np.arange(N))

# Grafika 1: PRE PRITISKA (Direktan sudar, divergencija divlja)
U_pre = (brzina_x[:, :-1] + brzina_x[:, 1:]) / 2.0
V_pre = (brzina_y[:-1, :] + brzina_y[1:, :]) / 2.0
ax1.quiver(X, Y, U_pre, -V_pre, color='red', scale=50)
ax1.set_title("PRE PRITISKA: Brzine se sudaraju i gužvaju")
ax1.set_xlim(-1, N)
ax1.set_ylim(N, -1)

# Grafika 2: POSLE PRITISKA (Fluid obilazi i skreće)
U_posle = (brzina_x_nova[:, :-1] + brzina_x_nova[:, 1:]) / 2.0
V_posle = (brzina_y_nova[:-1, :] + brzina_y_nova[1:, :]) / 2.0
# Pozadina prikazuje raspored pritiska (gde je sudar, pritisak je najveći - žuto)
ax2.imshow(P_matrica, cmap='jet', origin='upper')
ax2.quiver(X, Y, U_posle, -V_posle, color='white', scale=50)
ax2.set_title("POSLE PRITISKA: Pritisak (žuto) tera fluid da skrene")
ax2.set_xlim(-1, N)
ax2.set_ylim(N, -1)

plt.show()