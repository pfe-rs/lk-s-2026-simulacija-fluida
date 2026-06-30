import numpy as np
import cupy as cp
from inspect import signature
from cupyx.scipy import sparse as cp_sparse
from cupyx.scipy.sparse.linalg import cg as gpu_cg
from scipy.sparse import lil_matrix
from time import perf_counter


PRESETS = {
    "1": "single_vortex",
    "2": "double_vortex",
    "3": "four_vortices",
    "4": "shear_layer",
    "5": "taylor_green",
    "6": "blob",
}


class FluidSimulation:
    def __init__(
        self,
        n=128,
        dt=0.01,
        h=0.1,
        viscosity=0.08,
        density=1.0,
        preset="shear_layer",
        pressure_tol=1e-4,
        pressure_maxiter=None,
        dtype=cp.float32,
    ):
        self.n = n
        self.dt = dt
        self.h = h
        self.viscosity = viscosity
        self.density = density
        self.preset = preset
        self.pressure_tol = pressure_tol
        self.pressure_maxiter = pressure_maxiter
        self.dtype = dtype
        self.cpu_dtype = np.float32 if dtype == cp.float32 else np.float64
        self.cg_tolerance_name = "rtol" if "rtol" in signature(gpu_cg).parameters else "tol"
        self.frame = 0

        self.cell_type = cp.ones((n, n), dtype=cp.int32)
        self.cell_type[0, :] = 0
        self.cell_type[-1, :] = 0
        self.cell_type[:, 0] = 0
        self.cell_type[:, -1] = 0
        self.cell_type_cpu = cp.asnumpy(self.cell_type)

        self.index_map_cpu = self._build_index_map(self.cell_type_cpu)
        self.index_map = cp.asarray(self.index_map_cpu)
        self.fluid_mask = self.index_map != -1
        self.unknown_count = int(np.max(self.index_map_cpu) + 1)
        self.pressure_matrix = self._build_gpu_pressure_matrix()

        self.velocity_x = cp.zeros((n, n + 1), dtype=self.dtype)
        self.velocity_y = cp.zeros((n + 1, n), dtype=self.dtype)
        self.pressure = cp.zeros((n, n), dtype=self.dtype)
        self.x_i_grid, self.x_j_grid = cp.indices(self.velocity_x.shape)
        self.y_i_grid, self.y_j_grid = cp.indices(self.velocity_y.shape)

        self.reset(preset)

    def _sync_gpu(self):
        cp.cuda.Stream.null.synchronize()

    def _elapsed_ms(self, start):
        self._sync_gpu()
        return (perf_counter() - start) * 1000.0

    def _new_timings(self):
        return {
            "rhs_ms": 0.0,
            "pressure_ms": 0.0,
            "projection_ms": 0.0,
            "advection_ms": 0.0,
            "diffusion_ms": 0.0,
            "walls_ms": 0.0,
            "total_ms": 0.0,
        }

    def _build_index_map(self, cell_type):
        index_map = -np.ones(cell_type.shape, dtype=np.int32)
        fluid_mask = (cell_type != 0) & (cell_type != 2)
        index_map[fluid_mask] = np.arange(np.count_nonzero(fluid_mask), dtype=np.int32)
        return index_map

    def _build_gpu_pressure_matrix(self):
        matrix = lil_matrix((self.unknown_count, self.unknown_count), dtype=self.cpu_dtype)
        rows, columns = self.index_map_cpu.shape

        for i in range(rows):
            for j in range(columns):
                row_index = int(self.index_map_cpu[i, j])
                if row_index == -1:
                    continue

                neighbor_count = 0
                for ni, nj in ((i - 1, j), (i + 1, j), (i, j - 1), (i, j + 1)):
                    if self.cell_type_cpu[ni, nj] == 1:
                        neighbor_count += 1
                        matrix[row_index, int(self.index_map_cpu[ni, nj])] = 1.0
                    elif self.cell_type_cpu[ni, nj] == 2:
                        neighbor_count += 1

                matrix[row_index, row_index] = -float(neighbor_count)

        anchored = matrix.tolil()
        anchored[0, :] = 0.0
        anchored[:, 0] = 0.0
        anchored[0, 0] = -1.0

        return -cp_sparse.csr_matrix(anchored.tocsr())

    def reset(self, preset=None):
        if preset is not None:
            self.preset = preset

        if self.preset == "single_vortex":
            self.initialize_single_vortex()
        elif self.preset == "double_vortex":
            self.initialize_double_vortex()
        elif self.preset == "four_vortices":
            self.initialize_four_vortices()
        elif self.preset == "shear_layer":
            self.initialize_shear_layer(8.0)
        elif self.preset == "taylor_green":
            self.initialize_taylor_green()
        elif self.preset == "blob":
            self.initialize_blob()
        else:
            raise ValueError(f"Unknown preset: {self.preset}")

        self.pressure.fill(0.0)
        self.frame = 0

    def step(self, substeps=1, profile=False):
        if not profile:
            for _ in range(substeps):
                self._step_once()
            return None

        timings = self._new_timings()
        self._sync_gpu()
        total_start = perf_counter()

        for _ in range(substeps):
            self._step_once(timings)

        timings["total_ms"] = self._elapsed_ms(total_start)
        if substeps > 1:
            for key in timings:
                timings[key] /= substeps

        return timings

    def _step_once(self, timings=None):
        step_start = perf_counter() if timings is not None else None
        b_vector = self._pressure_rhs()
        if timings is not None:
            timings["rhs_ms"] += self._elapsed_ms(step_start)

        step_start = perf_counter() if timings is not None else None
        self.pressure = self._solve_pressure(b_vector)
        if timings is not None:
            timings["pressure_ms"] += self._elapsed_ms(step_start)

        step_start = perf_counter() if timings is not None else None
        self._project_pressure()
        if timings is not None:
            timings["projection_ms"] += self._elapsed_ms(step_start)

        step_start = perf_counter() if timings is not None else None
        advected_x = self._advect(self.velocity_x, self.velocity_x, self.velocity_y, "x_edge")
        advected_y = self._advect(self.velocity_y, self.velocity_x, self.velocity_y, "y_edge")
        if timings is not None:
            timings["advection_ms"] += self._elapsed_ms(step_start)

        step_start = perf_counter() if timings is not None else None
        self.velocity_x = self._diffuse(advected_x)
        self.velocity_y = self._diffuse(advected_y)
        if timings is not None:
            timings["diffusion_ms"] += self._elapsed_ms(step_start)

        step_start = perf_counter() if timings is not None else None
        self._enforce_walls()
        if timings is not None:
            timings["walls_ms"] += self._elapsed_ms(step_start)
        self.frame += 1

    def _pressure_rhs(self):
        b = cp.zeros(self.unknown_count, dtype=self.dtype)

        u_left = self.velocity_x[:, :-1].copy()
        u_right = self.velocity_x[:, 1:].copy()
        v_top = self.velocity_y[:-1, :].copy()
        v_bottom = self.velocity_y[1:, :].copy()

        u_left[:, 1:] = cp.where(self.cell_type[:, :-1] == 0, 0.0, u_left[:, 1:])
        u_right[:, :-1] = cp.where(self.cell_type[:, 1:] == 0, 0.0, u_right[:, :-1])
        v_top[1:, :] = cp.where(self.cell_type[:-1, :] == 0, 0.0, v_top[1:, :])
        v_bottom[:-1, :] = cp.where(self.cell_type[1:, :] == 0, 0.0, v_bottom[:-1, :])

        divergence = (u_right - u_left) + (v_bottom - v_top)
        b[self.index_map[self.fluid_mask]] = (self.density * self.h / self.dt) * divergence[
            self.fluid_mask
        ]
        return b

    def _solve_pressure(self, b_vector):
        anchored_b = b_vector.copy()
        anchored_b[0] = 0.0
        cg_options = {
            self.cg_tolerance_name: self.pressure_tol,
            "maxiter": self.pressure_maxiter,
        }
        pressure_vector, info = gpu_cg(
            self.pressure_matrix,
            -anchored_b,
            **cg_options,
        )
        if int(info) != 0:
            raise RuntimeError(f"GPU pressure solve did not converge, cg info={int(info)}")

        pressure = cp.zeros(self.cell_type.shape, dtype=self.dtype)
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

    def _advect(self, field, velocity_x, velocity_y, position_type):
        rows, columns = field.shape

        if position_type == "x_edge":
            i_grid = self.x_i_grid
            j_grid = self.x_j_grid
            u_wind = field
            i_top = cp.clip(i_grid - 1, 0, velocity_y.shape[0] - 1)
            i_bot = cp.clip(i_grid, 0, velocity_y.shape[0] - 1)
            j_left = cp.clip(j_grid - 1, 0, velocity_y.shape[1] - 1)
            j_right = cp.clip(j_grid, 0, velocity_y.shape[1] - 1)
            v_wind = (
                velocity_y[i_top, j_left]
                + velocity_y[i_bot, j_left]
                + velocity_y[i_top, j_right]
                + velocity_y[i_bot, j_right]
            ) / 4.0
        elif position_type == "y_edge":
            i_grid = self.y_i_grid
            j_grid = self.y_j_grid
            v_wind = field
            i_top = cp.clip(i_grid - 1, 0, velocity_x.shape[0] - 1)
            i_bot = cp.clip(i_grid, 0, velocity_x.shape[0] - 1)
            j_left = cp.clip(j_grid - 1, 0, velocity_x.shape[1] - 1)
            j_right = cp.clip(j_grid, 0, velocity_x.shape[1] - 1)
            u_wind = (
                velocity_x[i_top, j_left]
                + velocity_x[i_bot, j_left]
                + velocity_x[i_top, j_right]
                + velocity_x[i_bot, j_right]
            ) / 4.0
        else:
            raise ValueError(f"Unknown position type: {position_type}")

        i_old = cp.clip(i_grid - (v_wind * self.dt / self.h), 0.0, float(rows - 1))
        j_old = cp.clip(j_grid - (u_wind * self.dt / self.h), 0.0, float(columns - 1))

        i0 = cp.floor(i_old).astype(cp.int32)
        j0 = cp.floor(j_old).astype(cp.int32)
        i1 = cp.minimum(rows - 1, i0 + 1)
        j1 = cp.minimum(columns - 1, j0 + 1)

        beta = i_old - i0
        alpha = j_old - j0

        return (
            field[i0, j0] * (1 - alpha) * (1 - beta)
            + field[i0, j1] * alpha * (1 - beta)
            + field[i1, j0] * (1 - alpha) * beta
            + field[i1, j1] * alpha * beta
        )

    def _diffuse(self, field):
        new_field = cp.copy(field)
        laplacian = (
            field[:-2, 1:-1]
            + field[2:, 1:-1]
            + field[1:-1, :-2]
            + field[1:-1, 2:]
            - 4.0 * field[1:-1, 1:-1]
        ) / (self.h**2)
        new_field[1:-1, 1:-1] = field[1:-1, 1:-1] + self.viscosity * self.dt * laplacian
        return new_field

    def _enforce_walls(self):
        self.velocity_x[:, 0] = 0.0
        self.velocity_x[:, -1] = 0.0
        self.velocity_x[0, :] = 0.0
        self.velocity_x[-1, :] = 0.0
        self.velocity_y[0, :] = 0.0
        self.velocity_y[-1, :] = 0.0
        self.velocity_y[:, 0] = 0.0
        self.velocity_y[:, -1] = 0.0

    def add_vortex_at_cell(self, x_cell, y_cell, radius=0.5, strength=24.0):
        cx = float(np.clip(x_cell, 1, self.n - 2)) * self.h
        cy = float(np.clip(y_cell, 1, self.n - 2)) * self.h
        self._add_vortex(cx, cy, radius, strength)

    def _add_vortex(self, cx, cy, radius, strength):
        eps = 1e-8

        dx = self.x_j_grid * self.h - cx
        dy = (self.x_i_grid + 0.5) * self.h - cy
        r = cp.sqrt(dx * dx + dy * dy)
        mask = r < radius
        factor = cp.zeros_like(self.velocity_x)
        factor[mask] = (1.0 - cp.exp(-(r[mask] / radius) ** 2)) / (r[mask] + eps)
        self.velocity_x[mask] += -strength * dy[mask] * factor[mask]

        dx = (self.y_j_grid + 0.5) * self.h - cx
        dy = self.y_i_grid * self.h - cy
        r = cp.sqrt(dx * dx + dy * dy)
        mask = r < radius
        factor = cp.zeros_like(self.velocity_y)
        factor[mask] = (1.0 - cp.exp(-(r[mask] / radius) ** 2)) / (r[mask] + eps)
        self.velocity_y[mask] += strength * dx[mask] * factor[mask]

    def initialize_blob(self):
        self.velocity_x.fill(0.0)
        self.velocity_y.fill(0.0)

        center_i = self.n // 2
        center_j = self.n // 4
        radius = 5
        self.velocity_x[
            (self.x_i_grid - center_i) ** 2 + (self.x_j_grid - center_j) ** 2 < radius**2
        ] = 10.0
        self.velocity_y[
            (self.y_i_grid - center_i) ** 2 + (self.y_j_grid - center_j) ** 2 < radius**2
        ] = 10.0

    def initialize_single_vortex(self):
        self.velocity_x.fill(0.0)
        self.velocity_y.fill(0.0)
        self._add_vortex(self.n * self.h * 0.5, self.n * self.h * 0.5, 0.8, 40.0)

    def initialize_double_vortex(self):
        self.velocity_x.fill(0.0)
        self.velocity_y.fill(0.0)
        self._add_vortex(self.n * self.h * 0.30, self.n * self.h * 0.50, 0.6, 40.0)
        self._add_vortex(self.n * self.h * 0.70, self.n * self.h * 0.50, 0.6, -40.0)

    def initialize_four_vortices(self):
        self.velocity_x.fill(0.0)
        self.velocity_y.fill(0.0)
        for px, py, strength in (
            (0.30, 0.30, 40.0),
            (0.70, 0.30, -40.0),
            (0.30, 0.70, -40.0),
            (0.70, 0.70, 40.0),
        ):
            self._add_vortex(self.n * self.h * px, self.n * self.h * py, 0.45, strength)

    def initialize_shear_layer(self, strength):
        self.velocity_x.fill(0.0)
        self.velocity_y.fill(0.0)
        self.velocity_x[: self.n // 2, :] = strength
        self.velocity_x[self.n // 2 :, :] = -strength

    def initialize_taylor_green(self):
        self.velocity_x.fill(0.0)
        self.velocity_y.fill(0.0)

        u0 = 20.0
        length = self.n * self.h
        x = self.x_j_grid * self.h
        y = (self.x_i_grid + 0.5) * self.h
        self.velocity_x[:, :] = u0 * cp.sin(2 * cp.pi * x / length) * cp.cos(2 * cp.pi * y / length)

        x = (self.y_j_grid + 0.5) * self.h
        y = self.y_i_grid * self.h
        self.velocity_y[:, :] = -u0 * cp.cos(2 * cp.pi * x / length) * cp.sin(2 * cp.pi * y / length)

    def centered_velocity(self):
        u_center = (self.velocity_x[:, :-1] + self.velocity_x[:, 1:]) / 2.0
        v_center = (self.velocity_y[:-1, :] + self.velocity_y[1:, :]) / 2.0
        return u_center, v_center

    def speed(self):
        u_center, v_center = self.centered_velocity()
        return cp.sqrt(u_center**2 + v_center**2)

    def metrics(self):
        divergence = (
            self.velocity_x[1:-1, 2:self.n]
            - self.velocity_x[1:-1, 1 : self.n - 1]
            + self.velocity_y[2:self.n, 1:-1]
            - self.velocity_y[1 : self.n - 1, 1:-1]
        )

        vorticity = (
            (self.velocity_y[: self.n - 1, 1:self.n] - self.velocity_y[: self.n - 1, : self.n - 1])
            / self.h
            - (
                self.velocity_x[1:self.n, : self.n - 1]
                - self.velocity_x[: self.n - 1, : self.n - 1]
            )
            / self.h
        )

        return (
            float(cp.asnumpy(cp.sum(cp.abs(divergence)))),
            float(cp.asnumpy(cp.sum(cp.abs(vorticity)))),
        )
