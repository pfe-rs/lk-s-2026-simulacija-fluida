import numpy as np
import matplotlib.pyplot as plt
from essential_functions import Univerzalna_Advekcija, Viskoznost, Univerzalna_Difuzija


#GEMINI KOD NIJE MOJA VIZUALIZACIJA !!!! SAMO TRENUTNI TEST 
# ==========================================
# POSTAVKA SIMULACIJE VISKOZNOSTI
# ==========================================
N = 32
dt = 0.001  # Veoma mali dt da eksplicitna metoda ne eksplodira!
h = 0.1
ni = 0.5    # Koeficijent viskoznosti (unutrašnje trenje)

# Inicijalizacija
brzina_x = np.zeros((N, N+1))
brzina_y = np.zeros((N+1, N))

# Ubacujemo OŠTRU traku brzine na sredinu (kolone 15 i 16)
brzina_x[:, 14:17] = 10.0

X, Y = np.meshgrid(np.arange(N), np.arange(N))

plt.ion()
fig, ax = plt.subplots(figsize=(7, 7))

for frejm in range(80):
    ax.clear()
    
    # Primenjujemo tvoju ispravljenu univerzalnu difuziju
    brzina_x = Univerzalna_Difuzija(brzina_x, ni, dt, h)
    brzina_y = Univerzalna_Difuzija(brzina_y, ni, dt, h)
    
    # Svođenje na centar ćelije radi crtanja
    U_centar = (brzina_x[:, :-1] + brzina_x[:, 1:]) / 2.0
    V_centar = (brzina_y[:-1, :] + brzina_y[1:, :]) / 2.0
    brzina_magnituda = np.sqrt(U_centar**2 + V_centar**2)
    
    # Crtačke komande
    # Koristimo imshow da vidimo intenzitet brzine kao usijanje (cmap='hot')
    ax.imshow(U_centar, cmap='hot', origin='upper', extent=[-0.5, N-0.5, N-0.5, -0.5], vmin=0, vmax=10)
    
    # Preko toga crtamo strelice koje pokazuju jačinu
    ax.quiver(X, Y, U_centar, -V_centar, color='cyan', scale=50, width=0.003)
    
    ax.set_xlim(-0.5, N-0.5)
    ax.set_ylim(N-0.5, -0.5)
    ax.set_title(f"Viskoznost: Razlivanje oštre ivice (Frejm {frejm})")
    
    fig.canvas.draw_idle()
    fig.canvas.flush_events()
    plt.pause(0.01)

plt.ioff()
plt.show()