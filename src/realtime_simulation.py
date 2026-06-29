import numpy as np
import cupy as cp
from scipy.sparse.linalg import cg
from time import perf_counter
from scipy.sparse import lil_matrix, csr_matrix
from scipy.sparse.linalg import factorized
import metrics

PRESETS = {
    "1": "single_vortex",
    "2": "double_vortex",
    "3": "four_vortices",
    "4": "shear_layer",
    "5": "taylor_green",
    "6": "blob",
    "7" : "lid_driven_cavity"
}


class FluidSimulation:
    def __init__(
        self,
        n=32,
        dt=0.01,
        h=0.1,
        viscosity=0.08,
        density=1.0,
        preset="shear_layer",
    ):
        self.n = n
        self.dt = dt
        self.h = h
        self.viscosity = viscosity
        self.density = density
        self.preset = preset
        self.frame = 0

        self.cell_type = cp.ones((n, n), dtype=int)
        self.cell_type[0, :] = 0
        self.cell_type[-1, :] = 0
        self.cell_type[:, 0] = 0
        self.cell_type[:, -1] = 0
        #self.cell_type[n/2, ] = 0

        self.index_map = self.IndexMap(self.cell_type)
        self.system_matrix = self._build_sparse_system_matrix(self.index_map)
        self.pressure_solver = factorized(self._build_anchored_pressure_matrix())
        self.fluid_mask = self.index_map != -1

        self.velocity_x = cp.zeros((n, n + 1))
        self.velocity_y = cp.zeros((n + 1, n))
        self.pressure = cp.zeros((n, n))

        self.total_accuracy_sum = 0.0
        self.accuracy_frame_count = 0

        self.mouse_pressed_pos = None 
        self.mouse_timer = 0

        self.reset(preset)

    def Univerzalna_Advekcija(self, polje, brzina_x, brzina_y, tip_pozicije, dt=0.05, h=0.1):
        redova, kolone = polje.shape
        i_grid, j_grid = cp.indices(polje.shape)

        if tip_pozicije == 'x_ivica':
            u_vetar = polje
            i_top = cp.clip(i_grid - 1, 0, brzina_y.shape[0] - 1)
            i_bot = cp.clip(i_grid, 0, brzina_y.shape[0] - 1)
            j_left = cp.clip(j_grid - 1, 0, brzina_y.shape[1] - 1)
            j_right = cp.clip(j_grid, 0, brzina_y.shape[1] - 1)
            v_vetar = (
                brzina_y[i_top, j_left]
                + brzina_y[i_bot, j_left]
                + brzina_y[i_top, j_right]
                + brzina_y[i_bot, j_right]
            ) / 4.0
        elif tip_pozicije == 'y_ivica':
            v_vetar = polje
            i_top = cp.clip(i_grid - 1, 0, brzina_x.shape[0] - 1)
            i_bot = cp.clip(i_grid, 0, brzina_x.shape[0] - 1)
            j_left = cp.clip(j_grid - 1, 0, brzina_x.shape[1] - 1)
            j_right = cp.clip(j_grid, 0, brzina_x.shape[1] - 1)
            u_vetar = (
                brzina_x[i_top, j_left]
                + brzina_x[i_bot, j_left]
                + brzina_x[i_top, j_right]
                + brzina_x[i_bot, j_right]
            ) / 4.0
        else:
            raise ValueError(f"Nepoznat tip pozicije: {tip_pozicije}")

        i_staro = cp.clip(i_grid - (v_vetar * dt / h), 0.0, float(redova - 1))
        j_staro = cp.clip(j_grid - (u_vetar * dt / h), 0.0, float(kolone - 1))

        i0 = cp.floor(i_staro).astype(int)
        j0 = cp.floor(j_staro).astype(int)
        i1 = cp.minimum(redova - 1, i0 + 1)
        j1 = cp.minimum(kolone - 1, j0 + 1)

        beta = i_staro - i0
        alfa = j_staro - j0

        return (
            polje[i0, j0] * (1 - alfa) * (1 - beta)
            + polje[i0, j1] * alfa * (1 - beta)
            + polje[i1, j0] * (1 - alfa) * beta
            + polje[i1, j1] * alfa * beta
        )

    def Viskoznost(self, brzina_x, brzina_y, dt, h, ni):
        x_privremena = cp.zeros_like(brzina_x)
        y_privremena = cp.zeros_like(brzina_y)

        x_privremena[1:-1, 1:-1] = brzina_x[1:-1, 1:-1] + ni * dt * (
            brzina_x[:-2, 1:-1]
            + brzina_x[1:-1, :-2]
            + brzina_x[1:-1, 2:]
            + brzina_x[2:, 2:]
            - 4 * brzina_x[1:-1, 1:-1]
        ) / (h**2)

        y_privremena[1:-1, 1:-1] = brzina_y[1:-1, 1:-1] + ni * dt * (
            brzina_y[:-2, 1:-1]
            + brzina_y[1:-1, :-2]
            + brzina_y[1:-1, 2:]
            + brzina_y[2:, 2:]
            - 4 * brzina_y[1:-1, 1:-1]
        ) / (h**2)
        
        return x_privremena, y_privremena

    def Univerzalna_Difuzija(self, polje, ni, dt, h):

        novo_polje = cp.copy(polje) 
        laplasijan = (
            polje[:-2, 1:-1]
            + polje[2:, 1:-1]
            + polje[1:-1, :-2]
            + polje[1:-1, 2:]
            - 4.0 * polje[1:-1, 1:-1]
        ) / (h**2)
        novo_polje[1:-1, 1:-1] = polje[1:-1, 1:-1] + ni * dt * laplasijan
                
        return novo_polje

    def IndexMap(self, tip_celije):

        if hasattr(tip_celije, 'get'):
            tip_celije = tip_celije.get()
            
        tip_celije = np.asarray(tip_celije)
        mapa_indexa = -np.ones(tip_celije.shape, dtype=int)
        fluid_mask = (tip_celije != 0) & (tip_celije != 2)
        mapa_indexa[fluid_mask] = np.arange(np.count_nonzero(fluid_mask))
        
        return mapa_indexa
                    
    def Matrica_A(self, tip_celije, mapa_indexa):
        if hasattr(tip_celije, 'get'):
            tip_celije = tip_celije.get()

        tip_celije = np.array(tip_celije)

        N = int(np.max(mapa_indexa) + 1)
        
        
        matrica_a = lil_matrix((N, N), dtype=np.float64) 

        broj_redova, broj_kolona = mapa_indexa.shape

        for i in range(broj_redova):
            for j in range(broj_kolona):
                a = int(mapa_indexa[i, j])
                
                if a != -1:
                    cnt = 0 
                    
                    if tip_celije[i-1, j] == 1:
                        cnt += 1
                        matrica_a[a, int(mapa_indexa[i-1, j])] = 1.0 
                    if tip_celije[i+1, j] == 1:
                        cnt += 1
                        matrica_a[a, int(mapa_indexa[i+1, j])] = 1.0 
                    if tip_celije[i, j-1] == 1:
                        cnt += 1
                        matrica_a[a, int(mapa_indexa[i, j-1])] = 1.0 
                    if tip_celije[i, j+1] == 1:
                        cnt += 1
                        matrica_a[a, int(mapa_indexa[i, j+1])] = 1.0 

                    if tip_celije[i-1, j] == 2:
                        cnt += 1
                    if tip_celije[i+1, j] == 2:
                        cnt += 1
                    if tip_celije[i, j-1] == 2:
                        cnt += 1
                    if tip_celije[i, j+1] == 2:
                        cnt += 1

                    matrica_a[a, a] = -float(cnt)
                    
        return matrica_a # Vraća retku LIL matricu

    def vectorB(self, tip_celije, mapa_indexa, brzina_x, brzina_y, rho, dt, h):
        tip_celije = cp.array(tip_celije)
        
        N = int(cp.max(mapa_indexa) + 1)
        b = cp.zeros(N)

        u_levo = brzina_x[:, :-1].copy()
        u_desno = brzina_x[:, 1:].copy()
        v_gore = brzina_y[:-1, :].copy()
        v_dole = brzina_y[1:, :].copy()

        u_levo[:, 1:] = cp.where(tip_celije[:, :-1] == 0, 0.0, u_levo[:, 1:])
        u_desno[:, :-1] = cp.where(tip_celije[:, 1:] == 0, 0.0, u_desno[:, :-1])
        v_gore[1:, :] = cp.where(tip_celije[:-1, :] == 0, 0.0, v_gore[1:, :])
        v_dole[:-1, :] = cp.where(tip_celije[1:, :] == 0, 0.0, v_dole[:-1, :])

        divergencija = (u_desno - u_levo) + (v_dole - v_gore)
        fluid_mask = mapa_indexa != -1
        b[mapa_indexa[fluid_mask]] = (rho * h / dt) * divergencija[fluid_mask]
        return b

    def IzracunajPritisak(self, matrica_a, b_vektor, mapa_indexa, tip_celije, tol = 1e-5):
        
        P_vektor, _ = cg(matrica_a, b_vektor, rtol=tol)

        P_matrica = cp.zeros(tip_celije.shape)
        fluid_mask = mapa_indexa != -1
        P_matrica[fluid_mask] = P_vektor[mapa_indexa[fluid_mask].astype(int)]
                    
        return P_matrica

    def add_vortex(self, brzina_x, brzina_y, cx, cy, radius, strength, h):

        eps = 1e-8

        i_x, j_x = cp.indices(brzina_x.shape)
        dx = j_x * h - cx
        dy = (i_x + 0.5) * h - cy
        r = cp.sqrt(dx * dx + dy * dy)
        mask = r < radius
        factor = cp.zeros_like(brzina_x, dtype=float)
        factor[mask] = (1.0 - cp.exp(-(r[mask] / radius) ** 2)) / (r[mask] + eps)
        brzina_x[mask] += -strength * dy[mask] * factor[mask]

        i_y, j_y = cp.indices(brzina_y.shape)
        dx = (j_y + 0.5) * h - cx
        dy = i_y * h - cy
        r = cp.sqrt(dx * dx + dy * dy)
        mask = r < radius
        factor = cp.zeros_like(brzina_y, dtype=float)
        factor[mask] = (1.0 - cp.exp(-(r[mask] / radius) ** 2)) / (r[mask] + eps)
        brzina_y[mask] += strength * dx[mask] * factor[mask]

    def initialize_blob(self, brzina_x, brzina_y, N):

        brzina_x.fill(0)
        brzina_y.fill(0)

        centar_i = N // 2
        centar_j = N // 4
        radijus = 5

        i_x, j_x = cp.indices(brzina_x.shape)
        brzina_x[(i_x - centar_i) ** 2 + (j_x - centar_j) ** 2 < radijus**2] = 10

        i_y, j_y = cp.indices(brzina_y.shape)
        brzina_y[(i_y - centar_i) ** 2 + (j_y - centar_j) ** 2 < radijus**2] = 10

    def initialize_single_vortex(self, brzina_x, brzina_y, N, h):

        brzina_x.fill(0)
        brzina_y.fill(0)

        self.add_vortex(
            brzina_x,
            brzina_y,
            N*h*0.5,
            N*h*0.5,
            0.8,
            40,
            h
        )   

    def initialize_double_vortex(self, brzina_x, brzina_y, N, h):

        brzina_x.fill(0)
        brzina_y.fill(0)

        self.add_vortex(brzina_x, brzina_y,
                N*h*0.30,
                N*h*0.50,
                0.6,
                40,
                h)

        self.add_vortex(brzina_x, brzina_y,
                N*h*0.70,
                N*h*0.50,
                0.6,
                -40,
                h)

    def initialize_four_vortices(self, brzina_x, brzina_y, N, h):

        brzina_x.fill(0)
        brzina_y.fill(0)

        vortices = [
            (0.30,0.30,40),
            (0.70,0.30,-40),
            (0.30,0.70,-40),
            (0.70,0.70,40)
        ]

        for px,py,s in vortices:
            self.add_vortex(
                brzina_x,
                brzina_y,
                N*h*px,
                N*h*py,
                0.45,
                s,
                h
            )

    def initialize_shear_layer(self, brzina_x, brzina_y, strength, N):

        brzina_x.fill(0)
        brzina_y.fill(0)

        brzina_x[: N // 2, :] = strength
        brzina_x[N // 2 :, :] = -strength

    def initialize_taylor_green(self, brzina_x, brzina_y, N, h):

        brzina_x.fill(0)
        brzina_y.fill(0)

        U0 = 20
        L = N*h

        i_x, j_x = cp.indices(brzina_x.shape)
        x = j_x * h
        y = (i_x + 0.5) * h
        brzina_x[:, :] = U0 * cp.sin(2 * cp.pi * x / L) * cp.cos(2 * cp.pi * y / L)

        i_y, j_y = cp.indices(brzina_y.shape)
        x = (j_y + 0.5) * h
        y = i_y * h
        brzina_y[:, :] = -U0 * cp.cos(2 * cp.pi * x / L) * cp.sin(2 * cp.pi * y / L)

    def reset(self, preset=None):
        if preset is not None:
            self.preset = preset

        if self.preset == "single_vortex":
            self.initialize_single_vortex(self.velocity_x, self.velocity_y, self.n, self.h)
        elif self.preset == "double_vortex":
            self.initialize_double_vortex(self.velocity_x, self.velocity_y, self.n, self.h)
        elif self.preset == "four_vortices":
            self.initialize_four_vortices(self.velocity_x, self.velocity_y, self.n, self.h)
        elif self.preset == "shear_layer":
            self.initialize_shear_layer(self.velocity_x, self.velocity_y, 8, self.n)
        elif self.preset == "taylor_green":
            self.initialize_taylor_green(self.velocity_x, self.velocity_y, self.n, self.h)
        elif self.preset == "blob":
            self.initialize_blob(self.velocity_x, self.velocity_y, self.n)
        elif self.preset == "lid_driven_cavity":
            self.velocity_x.fill(0.0)
            self.velocity_y.fill(0.0)
        else:
            raise ValueError(f"Unknown preset: {self.preset}")

        self.total_accuracy_sum = 0.0
        self.accuracy_frame_count = 0

        self.pressure.fill(0.0)
        self.frame = 0

    def step(self, substeps=1, profile=False):
        timings = {
            "rhs_ms": 0.0,
            "pressure_ms": 0.0,
            "projection_ms": 0.0,
            "advection_ms": 0.0,
            "diffusion_ms": 0.0,
            "walls_ms": 0.0,
            "total_ms": 0.0,
        }
        total_start = perf_counter()

        for _ in range(substeps):
            step_start = perf_counter()
            
            b_vector = self._pressure_rhs()
            timings["rhs_ms"] += (perf_counter() - step_start) * 1000.0

            step_start = perf_counter()
            self.pressure = self._solve_pressure(b_vector)
            timings["pressure_ms"] += (perf_counter() - step_start) * 1000.0

            step_start = perf_counter()
            self._project_pressure()
            timings["projection_ms"] += (perf_counter() - step_start) * 1000.0

            self._enforce_walls()

            step_start = perf_counter()
            advected_x = self.Univerzalna_Advekcija(
                self.velocity_x,
                self.velocity_x,
                self.velocity_y,
                "x_ivica",
                self.dt,
                self.h,
            )
            advected_y = self.Univerzalna_Advekcija(
                self.velocity_y,
                self.velocity_x,
                self.velocity_y,
                "y_ivica",
                self.dt,
                self.h,
            )
            timings["advection_ms"] += (perf_counter() - step_start) * 1000.0

            step_start = perf_counter()
            self.velocity_x = self.Univerzalna_Difuzija(
                advected_x,
                self.viscosity,
                self.dt,
                self.h,
            )
            self.velocity_y = self.Univerzalna_Difuzija(
                advected_y,
                self.viscosity,
                self.dt,
                self.h,
            )
            timings["diffusion_ms"] += (perf_counter() - step_start) * 1000.0

            step_start = perf_counter()
            self._enforce_walls()

            current_acc = metrics.L2NormAcc(self.velocity_x, self.velocity_y)
            self.total_accuracy_sum += current_acc
            self.accuracy_frame_count += 1

            timings["walls_ms"] += (perf_counter() - step_start) * 1000.0
            self.frame += 1

        timings["total_ms"] = (perf_counter() - total_start) * 1000.0
        if substeps > 1:
            for key in timings:
                timings[key] /= substeps

        if profile:
            return timings
        return None

    def _build_sparse_system_matrix(self, mapa_indexa):
        
        retka_lil = self.Matrica_A(self.cell_type, mapa_indexa)
        
        retka_matrica = retka_lil.tocsr() 
        return retka_matrica
    
    def _pressure_rhs(self):

        return self.vectorB(
            self.cell_type,
            self.index_map,
            self.velocity_x, 
            self.velocity_y, 
            self.density, 
            self.dt,
            self.h
        )
    
    def _build_anchored_pressure_matrix(self):
        anchored = self.system_matrix.tolil(copy=True)
        anchored[0, :] = 0.0
        anchored[0, 0] = 1.0
        return anchored.tocsc()

    def _solve_pressure(self, b_vector):
        anchored_b = b_vector.copy()
        anchored_b[0] = 0.0
        
        # 1. Spuštamo vektor na CPU da bi SciPy mogao da ga reši
        anchored_b_cpu = anchored_b.get() 
        
        # 2. Rešavamo na CPU
        pressure_vector_cpu = self.pressure_solver(anchored_b_cpu) 
        
        # 3. Vraćamo rezultat nazad na GPU
        pressure_vector = cp.asarray(pressure_vector_cpu)

        pressure = cp.zeros(self.cell_type.shape)
        pressure[self.fluid_mask] = pressure_vector[self.index_map[self.fluid_mask]]
        pressure[self.fluid_mask] -= cp.mean(pressure[self.fluid_mask])
        return pressure

    def _project_pressure(self):
        pressure_scale = self.dt / (self.density * self.h)

        x_mask = (self.cell_type[:, 1:] != 0) & (self.cell_type[:, :-1] != 0)
        x_delta = pressure_scale * (self.pressure[:, 1:] - self.pressure[:, :-1])
        self.velocity_x[:, 1:self.n] = cp.where(
            x_mask,
            self.velocity_x[:, 1:self.n] - x_delta,
            0.0,
        )

        y_mask = (self.cell_type[1:, :] != 0) & (self.cell_type[:-1, :] != 0)
        y_delta = pressure_scale * (self.pressure[1:, :] - self.pressure[:-1, :])
        self.velocity_y[1:self.n, :] = cp.where(
            y_mask,
            self.velocity_y[1:self.n, :] - y_delta,
            0.0,
        )

    def _enforce_walls(self):
        self.velocity_x[:, 0] = 0.0
        self.velocity_x[:, -1] = 0.0
        self.velocity_x[0, :] = 0.0
        self.velocity_x[-1, :] = 0.0

        self.velocity_y[0, :] = 0.0
        self.velocity_y[-1, :] = 0.0
        self.velocity_y[:, 0] = 0.0
        self.velocity_y[:, -1] = 0.0

        if self.preset == "lid_driven_cavity":
            U0 = 0.0  # Možeš staviti fiksno ili izvući iz argumenata
            self.velocity_x[0, :] = U0

    def add_vortex_at_cell(self, x_cell, y_cell, radius=0.5, strength=24.0):

        cx = float(np.clip(x_cell, 1, self.n - 2)) * self.h
        cy = float(np.clip(y_cell, 1, self.n - 2)) * self.h
        self.add_vortex(self.velocity_x, self.velocity_y, cx, cy, radius, strength, self.h)

    def centered_velocity(self):
        u_center = (self.velocity_x[:, :-1] + self.velocity_x[:, 1:]) / 2.0
        v_center = (self.velocity_y[:-1, :] + self.velocity_y[1:, :]) / 2.0
        return u_center, v_center

    def speed(self):
        u_center, v_center = self.centered_velocity()
        return cp.sqrt(u_center**2 + v_center**2)

    def curl_field(self):
        curl = cp.zeros((self.n, self.n))
        curl[: self.n - 1, : self.n - 1] = (
            (self.velocity_y[: self.n - 1, 1:self.n] - self.velocity_y[: self.n - 1, : self.n - 1])
            / self.h
            - (
                self.velocity_x[1:self.n, : self.n - 1]
                - self.velocity_x[: self.n - 1, : self.n - 1]
            )
            / self.h
        )
        return curl

    def metrics(self):

        divergence = metrics.DivergenceMetric(self.velocity_x, self.velocity_y)

        vorticity, curl = metrics.Vorticity(self.velocity_x, self.velocity_y, self.h)

        kinetic_energy = metrics.KineticEnergy(self.velocity_x, self.velocity_y, self.density)

        cfl = metrics.CourantFriedrichLewy(self.velocity_x, self.velocity_y, self.dt, self.h)

        tacnost_L2 = metrics.L2NormAcc(self.velocity_x, self.velocity_y)

        avg_accuracy = self.total_accuracy_sum / self.accuracy_frame_count

        return divergence, vorticity, curl, kinetic_energy, cfl, tacnost_L2, avg_accuracy

    def curl_field(self):
    
        curl = cp.zeros((self.n, self.n))
        
        # Sređena sintaksa: dv_dx - du_dy
        curl[: self.n - 1, : self.n - 1] = (
            (self.velocity_y[: self.n - 1, 1 : self.n] - self.velocity_y[: self.n - 1, : self.n - 1]) / self.h
            - (self.velocity_x[1 : self.n, : self.n - 1] - self.velocity_x[: self.n - 1, : self.n - 1]) / self.h
        )
        return curl
    
    def curl_to_rgb(self, curl, curl_scale):
        
        normalized = cp.clip(curl / curl_scale, -1.0, 1.0)
        positive = cp.clip(normalized, 0.0, 1.0)
        negative = cp.clip(-normalized, 0.0, 1.0)

        rgb = cp.empty((*curl.shape, 3), dtype=cp.uint8)
        
        rgb[..., 0] = (35 + 220 * positive).astype(cp.uint8)
        rgb[..., 1] = (45 + 150 * (1.0 - cp.abs(normalized))).astype(cp.uint8)
        rgb[..., 2] = (55 + 200 * negative).astype(cp.uint8)
        return rgb
    
    def speed_to_rgb(self, speed, speed_scale):
        
        normalized = cp.clip(speed / speed_scale, 0.0, 1.0)
        
        rgb = cp.empty((*speed.shape, 3), dtype=cp.uint8)
        
        rgb[..., 0] = (255 * cp.minimum(normalized * 2.0, 1.0)).astype(cp.uint8)              # Crvena se pali odmah
        rgb[..., 1] = (255 * cp.clip((normalized - 0.3) * 2.0, 0.0, 1.0)).astype(cp.uint8)    # Zelena se pali kasnije (pravi narandžastu i žutu)
        rgb[..., 2] = (255 * cp.clip((normalized - 0.7) * 3.0, 0.0, 1.0)).astype(cp.uint8)    # Plava se pali na kraju za čisto bele "vruće" delove
        
        return rgb
    
    def apply_mouse_impulse(self, start_pos, end_pos, duration, M=5):
        
        # Pygame daje (X, Y), pa ih ovde jasno imenujemo:
        start_x, start_y = start_pos
        end_x, end_y = end_pos
        
        dx = end_x - start_x
        dy = end_y - start_y
        
        if dx == 0 and dy == 0:
            return
            
        multiplier = 80.0 
        vel_x_impulse = (dx / duration) * multiplier
        vel_y_impulse = (dy / duration) * multiplier
        
        
        half_m = M // 2
        
        
        r_start = max(1, start_y - half_m)
        r_end = min(self.n - 1, start_y + half_m + 1)
        c_start = max(1, start_x - half_m)
        c_end = min(self.n - 1, start_x + half_m + 1)
        
        
        
        self.velocity_x[r_start:r_end, c_start:c_end] += vel_x_impulse * 1.5 
        self.velocity_y[r_start:r_end, c_start:c_end] += vel_y_impulse * 1.5
        
        
        self._enforce_walls()
        
        
