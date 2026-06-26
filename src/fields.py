import numpy as np

class VectorField:
    def __init__(self, ime, x_grid, y_grid, u, v):
        self.ime = ime
        self.x_grid = x_grid
        self.y_grid = y_grid
        self.u = u
        self.v = v
    def Divergence(self) -> np.ndarray:

        dx = self.x_grid[0, 1] - self.x_grid[0, 0]
        dy = self.y_grid[1, 0] - self.y_grid[0, 0]
        
        dU_dx = np.gradient(self.u, axis=1) / dx
        dV_dy = np.gradient(self.v, axis=0) / dy

        return dU_dx + dV_dy
    
    def Value(self, i, j):
        return self.x_grid[i][j], self.y_grid[i][j]
    
    def Laplacian(self):
        dx = self.x_grid[0, 1] - self.x_grid[0, 0]
        dy = self.y_grid[1, 0] - self.y_grid[0, 0]

        dU_dy, dU_dx = np.gradient(self.u, dy, dx)

        d2U_dy, _ = np.gradient(dU_dy, dy, dx)
        _, d2U_dx = np.gradient(dU_dx, dy, dx)

        laplacian_u = d2U_dx + d2U_dy

        dV_dy, dV_dx = np.gradient(self.v, dy, dx)

        d2V_dy, _ = np.gradient(dV_dy, dy, dx)
        _, d2V_dx = np.gradient(dV_dx, dy, dx)

        laplacian_v = d2V_dx + d2V_dy

        return laplacian_u, laplacian_v
    
    def Laplacian(self):
 
        N = self.x_grid.shape[0]
        
        dx = self.x_grid[0, 1] - self.x_grid[0, 0]
        dy = self.y_grid[1, 0] - self.y_grid[0, 0]

        lap_u = np.zeros((N,N))
        lap_v = np.zeros((N,N))

        for i in range(N):
            for j in range(N):
                lap_u[i, j] = self.u[i, j+1] - 2*self.u[i, j] + self.u



    def MnozenjeSkalarom(self,skalar):
        novo_u = skalar * np.asarray(self.u)
        novo_v = skalar * np.asarray(self.v)

        return VectorField(f"Zbir ({self.ime} * skalar)", self.x_grid, self.y_grid, novo_u, novo_v)
        
    def SabiranjeVekPolja(self, drugo_polje):
        novo_u = self.u + drugo_polje.u
        novo_v = self.v + drugo_polje.v

        return VectorField(f"Zbir ({self.ime} + {drugo_polje.ime})", self.x_grid, self.y_grid, novo_u, novo_v)
    def Advekcija(self):
        
        U = np.asarray(self.u)
        V = np.asarray(self.v)

        dx = self.x_grid[0, 1] - self.x_grid[0, 0]
        dy = self.y_grid[1, 0] - self.y_grid[0, 0]

        dU_dy, dU_dx = np.gradient(U, dx, dy)
        dV_dy, dV_dx = np.gradient(V, dx, dy)

        advekcija_u = U * dU_dx + V*dU_dy
        advekcija_v = U * dV_dx + V * dV_dy

        return advekcija_u, advekcija_v



class ScalarField:
    
    def __init__(self, ime, vrednosti_grid, x_grid, y_grid):
        self.ime = ime
        self.vrednosti_grid = vrednosti_grid
        self.x_grid = x_grid
        self.y_grid = y_grid

    def Value(self, i, j):
        return self.vrednosti_grid[i][j], self.vrednosti_grid[i][j]

    def Gradient(self):

        dx = self.x_grid[0, 1] - self.x_grid[0, 0]
        dy = self.y_grid[1, 0] - self.y_grid[0, 0]

        N = self.vrednosti_grid.shape[0]

        grad_x = np.zeros((N+1, N+1))
        grad_y = np.zeros((N+1, N+1))

        #dp_dx = (pB + pC - pD - pA) / 2dx
        #dp_dy = (pD + pC - pB - pA) / 2dy
        
        for i in range(1, N):
            for j in range(1, N):
                grad_x[i, j] = (
                    (self.vrednosti_grid[i, j] + self.vrednosti_grid[i - 1, j]) - (self.vrednosti_grid[i, j - 1] + self.vrednosti_grid[i - 1, j - 1])
                ) / (2.0 * dx)
                grad_y[i, j] = (self.vrednosti_grid[i-1, j-1] + self.vrednosti_grid[i-1, j] - self.vrednosti_grid[i, j] - self.vrednosti_grid[i, j-1]) / (2.0 * dy)
        
        return grad_y, grad_x