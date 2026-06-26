##OVAJ KOD POKRECE CELU SIMULACIJU
import numpy as np
import matplotlib.pyplot as plt
from essential_functions import Univerzalna_Advekcija

# ==========================================
# ČISTA FOR PETLJA BEZ ZAVISNOSTI OD ANIMATION MODULA
# ==========================================
#GEMINI KOD NIJE MOJA VIZUALIZACIJA !!!! SAMO TRENUTNI TEST 
N = 20
dt = -0.005
h = 0.1

brzina_x = np.zeros((N, N+1))
brzina_y = np.zeros((N+1, N))

# Vrtlog
brzina_x[6:12, 5:11] = 4.0   
brzina_x[12:16, 9:15] = -4.0 
brzina_y[5:11, 12:16] = 4.0   
brzina_y[11:15, 5:11] = -4.0  

X, Y = np.meshgrid(np.arange(N), np.arange(N))

plt.ion() 
fig, ax = plt.subplots(figsize=(6, 6))

for frejm in range(50):
    ax.clear()
    
    # Korak advekcije
    bx_nova = Univerzalna_Advekcija(brzina_x, brzina_x, brzina_y, 'x_ivica', dt, h)
    by_nova = Univerzalna_Advekcija(brzina_y, brzina_x, brzina_y, 'y_ivica', dt, h)
    
    brzina_x = bx_nova
    brzina_y = by_nova
    
    U_centar = (brzina_x[:, :-1] + brzina_x[:, 1:]) / 2.0
    V_centar = (brzina_y[:-1, :] + brzina_y[1:, :]) / 2.0
    brzina_magnituda = np.sqrt(U_centar**2 + V_centar**2)
    
    ax.quiver(X, Y, U_centar, -V_centar, brzina_magnituda, cmap='jet', scale=40, width=0.005)
    ax.set_xlim(-1, N)
    ax.set_ylim(N, -1)
    ax.set_title(f"Advekcija brzine - Frejm {frejm}")
    
    fig.canvas.draw_idle()
    fig.canvas.flush_events()
    plt.pause(0.02)

plt.ioff()
plt.show()