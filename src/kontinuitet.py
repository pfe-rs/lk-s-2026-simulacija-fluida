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
        geometry="venturi",
    ):
        self.n = n
        self.dt = dt
        self.h = h
        self.viscosity = viscosity
        self.density = density
        self.preset = preset
        self.geometry = geometry
        self.frame = 0

        self.cell_type = cp.ones((n, n), dtype=int)
        
        self.cell_type[0, :] = 0
        self.cell_type[-1, :] = 0

        
        x1 = n // 3       
        x2 = 2 * n // 3   

        
        y_gore_siroko = n // 4       
        y_dole_siroko = 3 * n // 4    
        
        y_gore_usko = int(n // 2.2)  
        y_dole_usko = int(n // 1.8)  

        self.cell_type[y_gore_siroko, :x1] = 0
        self.cell_type[y_dole_siroko, :x1] = 0
        
        
        x_indices = cp.arange(x1, x2)
        dx = x2 - x1
        
        
        y_gornja = y_gore_siroko + (x_indices - x1) * (y_gore_usko - y_gore_siroko) // dx
        self.cell_type[y_gornja, x_indices] = 0
        self.cell_type[y_gornja, cp.clip(x_indices + 1, 0, n - 1)] = 0 # Ojačanje

        
        y_donja = y_dole_siroko + (x_indices - x1) * (y_dole_usko - y_dole_siroko) // dx
        self.cell_type[y_donja, x_indices] = 0
        self.cell_type[y_donja, cp.clip(x_indices + 1, 0, n - 1)] = 0 # Ojačanje

        
        self.cell_type[y_gore_usko, x2:n] = 0
        self.cell_type[y_dole_usko, x2:n] = 0

        self.cell_type[:y_gore_siroko, 0] = 0
        self.cell_type[y_dole_siroko:, 0] = 0
        

        self.cell_type[:y_gore_usko, -1] = 0
        self.cell_type[y_dole_usko:, -1] = 0

        self.inlet_speed = 10.0
        self.cell_type = cp.zeros((n, n), dtype=int)
        self._carve_duct()
        self._build_velocity_boundary_masks()

        self.index_map = self.IndexMap(self.cell_type)
        self.system_matrix = self._build_sparse_system_matrix(self.index_map)
        self.pressure_dirichlet_indices = self._pressure_dirichlet_indices()
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

    def _carve_duct(self):
        if self.geometry == "venturi":
            self._carve_venturi_duct()
        elif self.geometry == "flat":
            self._carve_flat_duct()
        else:
            raise ValueError(f"Unknown duct geometry: {self.geometry}")

    def _carve_venturi_duct(self):
        x1 = self.n // 5
        x2 = 7 * self.n // 10
        self.wide_sample_column = max(0, x1 // 2)
        self.narrow_sample_column = min(self.n - 1, x2 + max(1, self.n - x2) // 2)

        wide_half_width = self.n // 4
        narrow_half_width = max(2, self.n // 16)
        center = self.n // 2

        y_top_wide = center - wide_half_width
        y_bottom_wide = self.n - y_top_wide
        y_top_narrow = center - narrow_half_width
        y_bottom_narrow = self.n - y_top_narrow

        self.cell_type[y_top_wide:y_bottom_wide, :x1] = 1
        self.cell_type[y_top_narrow:y_bottom_narrow, x2:] = 1

        dx = max(1, x2 - x1)
        for x in range(x1, x2 + 1):
            t = x - x1
            y_top = y_top_wide + (t * (y_top_narrow - y_top_wide) + dx // 2) // dx
            y_bottom = self.n - y_top
            self.cell_type[y_top:y_bottom, x] = 1

    def _carve_flat_duct(self):
        wide_half_width = self.n // 4
        center = self.n // 2

        y_top = center - wide_half_width
        y_bottom = self.n - y_top
        self.cell_type[y_top:y_bottom, :] = 1

        self.wide_sample_column = self.n // 4
        self.narrow_sample_column = 3 * self.n // 4

    def _build_velocity_boundary_masks(self):
        self.velocity_x_fluid_mask = cp.zeros((self.n, self.n + 1), dtype=bool)
        self.velocity_x_fluid_mask[:, 1:self.n] = (
            (self.cell_type[:, :-1] == 1) & (self.cell_type[:, 1:] == 1)
        )
        self.inlet_open_mask = self.cell_type[:, 0] == 1
        self.inlet_drive_mask = self.inlet_open_mask.copy()
        inlet_rows = cp.where(self.inlet_open_mask)[0]
        if inlet_rows.size > 2:
            self.inlet_drive_mask[inlet_rows[0]] = False
            self.inlet_drive_mask[inlet_rows[-1]] = False
        self.inlet_lip_mask = self.inlet_open_mask & ~self.inlet_drive_mask
        self.outlet_mask = self.cell_type[:, -1] == 1
        self.velocity_x_fluid_mask[self.inlet_open_mask, 0] = True
        self.velocity_x_fluid_mask[self.outlet_mask, -1] = True

        self.velocity_y_fluid_mask = cp.zeros((self.n + 1, self.n), dtype=bool)
        self.velocity_y_fluid_mask[1:self.n, :] = (
            (self.cell_type[:-1, :] == 1) & (self.cell_type[1:, :] == 1)
        )

    def _pressure_dirichlet_indices(self):
        outlet_indices = self.index_map[self.cell_type.get()[:, -1] == 1, -1]
        outlet_indices = outlet_indices[outlet_indices != -1]
        if outlet_indices.size > 0:
            return outlet_indices.astype(int)
        return np.array([0], dtype=int)

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
                    

                    if i > 0:
                        if tip_celije[i-1, j] == 1:
                            cnt += 1
                            matrica_a[a, int(mapa_indexa[i-1, j])] = 1.0 
                        elif tip_celije[i-1, j] == 2:
                            cnt += 1


                    if i < broj_redova - 1:
                        if tip_celije[i+1, j] == 1:
                            cnt += 1
                            matrica_a[a, int(mapa_indexa[i+1, j])] = 1.0 
                        elif tip_celije[i+1, j] == 2:
                            cnt += 1

                    if j > 0:
                        if tip_celije[i, j-1] == 1:
                            cnt += 1
                            matrica_a[a, int(mapa_indexa[i, j-1])] = 1.0 
                        elif tip_celije[i, j-1] == 2:
                            cnt += 1


                    if j < broj_kolona - 1:
                        if tip_celije[i, j+1] == 1:
                            cnt += 1
                            matrica_a[a, int(mapa_indexa[i, j+1])] = 1.0 
                        elif tip_celije[i, j+1] == 2:
                            cnt += 1

                    matrica_a[a, a] = -float(cnt)
                    
        return matrica_a
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
        radijus = 20

        i_x, j_x = cp.indices(brzina_x.shape)

        #elf.velocity_x[:, 25] = 1

        i_y, j_y = cp.indices(brzina_y.shape)
        brzina_y[(i_y - centar_i) ** 2 + (j_y - centar_j) ** 2 < radijus**2] = 0

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

        self._enforce_walls()

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

            #self.density[:, 10] = 1.0

            current_acc = metrics.L2NormAcc(self.velocity_x, self.velocity_y)
            self.total_accuracy_sum += current_acc
            self.accuracy_frame_count += 1

            timings["walls_ms"] += (perf_counter() - step_start) * 1000.0
            self.frame += 1

        timings["total_ms"] = (perf_counter() - total_start) * 1000.0
        if substeps > 1:
            for key in timings:
                timings[key] /= substeps


        #fiksno_x = -30
        #self.velocity_x[:, fiksno_x] += 0.5

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
        for index in self.pressure_dirichlet_indices:
            anchored[index, :] = 0.0
            anchored[index, index] = 1.0
        return anchored.tocsc()

    def _solve_pressure(self, b_vector):
        anchored_b = b_vector.copy()
        anchored_b[self.pressure_dirichlet_indices] = 0.0
        
        anchored_b_cpu = anchored_b.get() 
        
        pressure_vector_cpu = self.pressure_solver(anchored_b_cpu) 
        

        pressure_vector = cp.asarray(pressure_vector_cpu)

        pressure = cp.zeros(self.cell_type.shape)
        pressure[self.fluid_mask] = pressure_vector[self.index_map[self.fluid_mask]]
        pressure[self.fluid_mask] -= cp.mean(pressure[self.fluid_mask])
        return pressure

    def _project_pressure(self):
        pressure_scale = self.dt / (self.density * self.h)

        x_mask = (self.cell_type[:, :-1] != 0) & (self.cell_type[:, 1:] != 0)
        

        x_delta = pressure_scale * (self.pressure[:, 1:] - self.pressure[:, :-1])
        

        self.velocity_x[:, 1:-1] = cp.where(
            x_mask,
            self.velocity_x[:, 1:-1] - x_delta,
            0.0
        )

        ulazni_vrat = self.inlet_drive_mask
        self.velocity_x[self.inlet_lip_mask, 0] = 0.0
        self.velocity_x[self.inlet_lip_mask, 1] = 0.0
        self.velocity_x[ulazni_vrat, 0] = self.inlet_speed
        self.velocity_x[ulazni_vrat, 1] = self.inlet_speed

        y_mask = (self.cell_type[:-1, :] != 0) & (self.cell_type[1:, :] != 0)
        

        y_delta = pressure_scale * (self.pressure[1:, :] - self.pressure[:-1, :])
        
        self.velocity_y[1:-1, :] = cp.where(
            y_mask,
            self.velocity_y[1:-1, :] - y_delta,
            0.0
        )

    def _enforce_walls(self):
        self.velocity_x = cp.where(self.velocity_x_fluid_mask, self.velocity_x, 0.0)
        self.velocity_y = cp.where(self.velocity_y_fluid_mask, self.velocity_y, 0.0)

        ulazni_vrat = self.inlet_drive_mask
        self.velocity_x[self.inlet_lip_mask, 0] = 0.0
        self.velocity_x[self.inlet_lip_mask, 1] = 0.0
        self.velocity_x[ulazni_vrat, 0] = self.inlet_speed
        self.velocity_x[ulazni_vrat, 1] = self.inlet_speed

        izlazni_vrat = (self.cell_type[:, -1] == 1) 
        self.velocity_x[izlazni_vrat, -1] = self.velocity_x[izlazni_vrat, -2]
        self.velocity_x[izlazni_vrat, -1] = cp.maximum(self.velocity_x[izlazni_vrat, -1], 0.0)

        if self.preset == "lid_driven_cavity":
            U0 = 0.0  
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

    def mean_speed_at_column(self, column):
        column = int(np.clip(column, 0, self.n - 1))
        fluid_rows = self.cell_type[:, column] == 1
        if not bool(cp.any(fluid_rows).get()):
            return 0.0
        return float(cp.mean(self.speed()[fluid_rows, column]).get())

    def speed_samples(self):
        speed_wide = self.mean_speed_at_column(self.wide_sample_column)
        speed_narrow = self.mean_speed_at_column(self.narrow_sample_column)
        if abs(speed_wide) > 1e-8:
            speed_ratio = speed_narrow / speed_wide
        else:
            speed_ratio = 0.0
        return speed_wide, speed_narrow, speed_ratio

    def mean_pressure_at_column(self, column):
        column = int(np.clip(column, 0, self.n - 1))
        fluid_rows = self.cell_type[:, column] == 1
        if not bool(cp.any(fluid_rows).get()):
            return 0.0
        return float(cp.mean(self.pressure[fluid_rows, column]).get())

    def pressure_samples(self):
        p_wide = self.mean_pressure_at_column(self.wide_sample_column)
        p_narrow = self.mean_pressure_at_column(self.narrow_sample_column)
        return p_wide, p_narrow, p_wide - p_narrow

    def profile_samples(self):
        u_center, _ = self.centered_velocity()
        speed = self.speed()
        x = np.arange(self.n)
        u_centerline = cp.asnumpy(u_center[self.n // 2, :])
        pressure_mean = np.zeros(self.n)
        speed_mean = np.zeros(self.n)

        for column in range(self.n):
            fluid_rows = self.cell_type[:, column] == 1
            if bool(cp.any(fluid_rows).get()):
                pressure_mean[column] = float(cp.mean(self.pressure[fluid_rows, column]).get())
                speed_mean[column] = float(cp.mean(speed[fluid_rows, column]).get())

        return {
            "x": x,
            "velocity_x": u_centerline,
            "pressure": pressure_mean,
            "speed": speed_mean,
        }

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

        p_wide, p_narrow, p_delta = self.pressure_samples()
        speed_wide, speed_narrow, speed_ratio = self.speed_samples()

        return (
            divergence,
            vorticity,
            curl,
            kinetic_energy,
            cfl,
            tacnost_L2,
            avg_accuracy,
            p_wide,
            p_narrow,
            p_delta,
            speed_wide,
            speed_narrow,
            speed_ratio,
        )

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
        
        rgb[..., 0] = (255 * cp.minimum(normalized * 2.0, 1.0)).astype(cp.uint8)              
        rgb[..., 1] = (255 * cp.clip((normalized - 0.3) * 2.0, 0.0, 1.0)).astype(cp.uint8)    
        rgb[..., 2] = (255 * cp.clip((normalized - 0.7) * 3.0, 0.0, 1.0)).astype(cp.uint8)    
        
        return rgb
    
    def apply_mouse_impulse(self, start_pos, end_pos, duration, M=5):
        

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
        
        
