import numpy as np
import matplotlib.pyplot as plt
from essential_functions import Univerzalna_Advekcija, Univerzalna_Difuzija
import matplotlib.animation as animation

#GEMINI KOD NIJE MOJA VIZUALIZACIJA !!!! SAMO TRENUTNI TEST 
# ==========================================
# SIMULACIJA: ADVEKCIJA + DIFUZIJA ZAJEDNO!
# ==========================================
N = 32
dt = 0.01 # Držimo mali dt zbog stabilnosti eksplicitne difuzije
h = 0.1
ni = 0.2    # Viskoznost

# Inicijalizacija polja
brzina_x = np.zeros((N, N+1))
brzina_y = np.zeros((N+1, N))

# Inicijalni blob brzine u donjem levom uglu
centar_i, centar_j = N - 8, 8
radijus = 4

for i in range(brzina_x.shape[0]):
    for j in range(brzina_x.shape[1]):
        if (i - centar_i)**2 + (j - centar_j)**2 < radijus**2:
            brzina_x[i, j] = 8.0

for i in range(brzina_y.shape[0]):
    for j in range(brzina_y.shape[1]):
        if (i - centar_i)**2 + (j - centar_j)**2 < radijus**2:
            brzina_y[i, j] = -8.0

X, Y = np.meshgrid(np.arange(N), np.arange(N))

plt.ion()
fig, ax = plt.subplots(figsize=(7, 7))

for frejm in range(100):
    ax.clear()
    
    # ---------------------------------------------------------
    # KORAK 1: ADVEKCIJA (Brzina pomera samu sebe kroz prostor)
    # ---------------------------------------------------------
    bx_advektovano = Univerzalna_Advekcija(brzina_x, brzina_x, brzina_y, 'x_ivica', dt, h)
    by_advektovano = Univerzalna_Advekcija(brzina_y, brzina_x, brzina_y, 'y_ivica', dt, h)
    
    # ---------------------------------------------------------
    # KORAK 2: DIFUZIJA (Viskoznost deluje na advektovano polje)
    # ---------------------------------------------------------
    brzina_x = Univerzalna_Difuzija(bx_advektovano, ni, dt, h)
    brzina_y = Univerzalna_Difuzija(by_advektovano, ni, dt, h)
    
    # ---------------------------------------------------------
    # VIZUELIZACIJA
    # ---------------------------------------------------------
    U_centar = (brzina_x[:, :-1] + brzina_x[:, 1:]) / 2.0
    V_centar = (brzina_y[:-1, :] + brzina_y[1:, :]) / 2.0
    brzina_magnituda = np.sqrt(U_centar**2 + V_centar**2)
    
    ax.imshow(brzina_magnituda, cmap='hot', origin='upper', extent=[-0.5, N-0.5, N-0.5, -0.5], vmin=0, vmax=12)
    ax.quiver(X, Y, U_centar, -V_centar, color='cyan', scale=120, width=0.003)
    
    ax.set_xlim(-0.5, N-0.5)
    ax.set_ylim(N-0.5, -0.5)
    ax.set_title(f"Advekcija + Difuzija | Frejm {frejm}")
    
    fig.canvas.draw_idle()
    fig.canvas.flush_events()
    plt.pause(0.001)

plt.ioff()
plt.show()