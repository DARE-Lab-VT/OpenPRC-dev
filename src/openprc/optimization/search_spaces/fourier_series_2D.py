import numpy as np
import matplotlib.pyplot as plt

from mpl_toolkits.mplot3d import Axes3D

np.set_printoptions(linewidth=240)


class FourierSeries2D:
    def __init__(self, rows, cols):
        self.domain_size = (1, 1) # 1m
        self.fine_grid_points_x = cols*20 + 1  # fine resolution for continuous function
        self.fine_grid_points_y = rows*20 + 1
        self.x_fine = np.linspace(0, self.domain_size[0], self.fine_grid_points_x)
        self.y_fine = np.linspace(0, self.domain_size[1], self.fine_grid_points_y)
        self.X_fine, self.Y_fine = np.meshgrid(self.x_fine, self.y_fine)

        self.coarse_grid_points_x = cols + (cols-1)*1 + 1
        self.coarse_grid_points_y = rows + (rows-1)*1 + 1

        # Number of Fourier basis
        self.N = 5 
        self.M = 5

        # Natural wavelength (Set it to domain size)
        self.lambda_x = self.domain_size[0]
        self.lambda_y = self.domain_size[1]

    def generate_fourier_coeff(self):
        np.random.seed(42)
        return np.random.uniform(-1, 1, self.N*self.M*4)

    def generate_fourier_series(self, fourier_coeff):
        F = np.zeros_like(self.X_fine)
        index = 0
        for n in range(self.N):
            for m in range(self.M):
                alpha = fourier_coeff[index]
                beta = fourier_coeff[index + 1]
                gamma = fourier_coeff[index + 2]
                delta = fourier_coeff[index + 3]
                index += 4
                A = 2*np.pi*n*self.X_fine / self.lambda_x
                B = 2*np.pi*m*self.Y_fine / self.lambda_y
                F += alpha * np.cos(A) * np.cos(B)
                F += beta * np.sin(A) * np.cos(B)
                F += gamma * np.cos(A) * np.sin(B)
                F += delta * np.sin(A) * np.sin(B)
        return F

    def discretize_fourier_series(self, F):
        block_size_x = F.shape[1] // (self.coarse_grid_points_x - 1)  # Determine block size for averaging
        block_size_y = F.shape[0] // (self.coarse_grid_points_y - 1)  # Determine block size for averaging
        F_discr = np.zeros((self.coarse_grid_points_y - 1, self.coarse_grid_points_x - 1))
        for r in range(self.coarse_grid_points_y - 1):
            for c in range(self.coarse_grid_points_x - 1):
                start_x = c * block_size_x
                end_x = start_x + block_size_x
                start_y = r * block_size_y
                end_y = start_y + block_size_y
                # Compute the mean value within the coarse cell
                F_discr[r, c] = np.mean(F[start_y:end_y, start_x:end_x])

        # --- Min-Max Normalization (range from 0 to 1)
        mn = F_discr.min()
        mx = F_discr.max()
        if mx == mn:
            F_discr_norm = np.zeros_like(F_discr)   # all same value -> set to 0
        else:
            F_discr_norm = (F_discr - mn) / (mx - mn)

        return F_discr_norm
    
    def check_global_connectivity(self, A, tol = 1e-12):
        A = np.asarray(A, dtype=float)

        # --- basic shape check ---
        if A.ndim != 2 or A.shape[0] != A.shape[1]:
            raise ValueError(f"Adjacency matrix must be square, got shape {A.shape}")

        # --- strict symmetry check ---
        if not np.allclose(A, A.T, atol=tol):
            raise ValueError("Adjacency matrix is not symmetric. "
                            "Undirected graph assumption violated.")

        # --- no self-loops ---
        if np.trace(A) != 0:
            raise ValueError("Adjacency matrix diagonal must be exactly all zeros (no self loop).")

        n = A.shape[0]
        if n <= 1:
            return True, np.array([0.0])

        # --- Laplacian ---
        D = np.diag(A.sum(axis=1))
        L = D - A

        # --- eigenvalues ---
        eigvals = np.linalg.eigvalsh(L)
        eigvals.sort()

        connected = eigvals[1] > tol
        return connected, eigvals

    def build_adjacency_matrix(self, F_discr_norm: np.ndarray, rows: int, cols: int, t_hi: float = 0.67, t_mid: float = 0.33, rigid_outer_frame: bool = True):
        F_discr_norm = np.asarray(F_discr_norm, dtype=float)

        expected_shape = (rows + (rows-1)*1, cols + (cols-1)*1)
        if F_discr_norm.shape != expected_shape:
            raise ValueError(f"F_norm must have shape {expected_shape}, got {F_discr_norm.shape}")

        n = rows * cols
        A_new = np.zeros((n, n), dtype=int)
        K_new = np.zeros((n, n), dtype=float)

        def _nid(r, c): return r * cols + c

        def _stiffness_from_mean(m):
            if m >= t_hi:
                return 222.15
            elif m >= t_mid:
                return 195.64
            return 0.0
        
        def _is_outer_frame_edge(r, c, rr, cc):
            # Only orthogonal neighbors
            dr = rr - r
            dc = cc - c
            if abs(dr) + abs(dc) != 1:
                return False

            # horizontal edge on top/bottom row
            if dr == 0 and (r == 0 or r == rows - 1):
                return True

            # vertical edge on left/right col
            if dc == 0 and (c == 0 or c == cols - 1):
                return True

            return False

        for r in range(rows):
            for c in range(cols):
                i = _nid(r, c)

                # 4-neighbors + 4 diagonals (original-grid neighbors)
                # directions = [
                #     (-1, -1), (-1, 0), (-1, 1),
                #     ( 0, -1),          ( 0, 1),
                #     ( 1, -1), ( 1, 0), ( 1, 1),
                # ]
                directions = [
                    (-1, 0),
                    ( 0, -1), ( 0, 1),
                    ( 1, 0),
                ]
                for dr, dc in directions:
                    rr, cc = r + dr, c + dc
                    if not (0 <= rr < rows and 0 <= cc < cols):
                        continue
                    j = _nid(rr, cc)
                    if j <= i:
                        continue

                    if rigid_outer_frame and _is_outer_frame_edge(r, c, rr, cc):
                        A_new[i, j] = A_new[j, i] = 1
                        K_new[i, j] = K_new[j, i] = -1
                        continue

                    # node cell indices (cell-center indices)
                    cy_i, cx_i = 2*r, 2*c
                    cy_j, cx_j = 2*rr, 2*cc

                    # midpoint cell index (exactly between)
                    cy_m = (cy_i + cy_j) // 2
                    cx_m = (cx_i + cx_j) // 2

                    mean_val = (F_discr_norm[cy_i, cx_i] + F_discr_norm[cy_m, cx_m] + F_discr_norm[cy_j, cx_j]) / 3.0
                    kij = _stiffness_from_mean(mean_val)

                    if kij > 0:
                        A_new[i, j] = A_new[j, i] = 1
                        K_new[i, j] = K_new[j, i] = kij

        # for r in range(rows - 1):
        #     for c in range(cols - 1):

        #         # "\" diagonal endpoints
        #         r1, c1 = r, c
        #         r2, c2 = r + 1, c + 1

        #         # "/" diagonal endpoints
        #         r3, c3 = r, c + 1
        #         r4, c4 = r + 1, c

        #         # helper to compute mean along an edge using your exact indexing scheme
        #         def edge_mean(rr_a, cc_a, rr_b, cc_b):
        #             cy_a, cx_a = 2 * rr_a, 2 * cc_a
        #             cy_b, cx_b = 2 * rr_b, 2 * cc_b
        #             cy_m = (cy_a + cy_b) // 2
        #             cx_m = (cx_a + cx_b) // 2
        #             return (F_discr_norm[cy_a, cx_a] + F_discr_norm[cy_m, cx_m] + F_discr_norm[cy_b, cx_b]) / 3.0

        #         mean_back = edge_mean(r1, c1, r2, c2)  # "\"
        #         mean_fwd  = edge_mean(r3, c3, r4, c4)  # "/"

        #         k_back = _stiffness_from_mean(mean_back)
        #         k_fwd  = _stiffness_from_mean(mean_fwd)

        #         # keep only the better diagonal (tie-breaker: prefer "\")
        #         if (k_back == 0) or (k_fwd == 0):
        #             continue

        #         use_back = (mean_back >= mean_fwd)

        #         if use_back and k_back > 0:
        #             i = _nid(r3, c3)
        #             j = _nid(r4, c4)
        #             A_new[i, j] = A_new[j, i] = 0
        #             K_new[i, j] = K_new[j, i] = 0.0

        #         elif (not use_back) and k_fwd > 0:
        #             i = _nid(r1, c1)
        #             j = _nid(r2, c2)
        #             A_new[i, j] = A_new[j, i] = 0
        #             K_new[i, j] = K_new[j, i] = 0.0

        np.fill_diagonal(A_new, 0)
        np.fill_diagonal(K_new, 0.0)
        return A_new, K_new
    
    def build_full_neighbor_topology(self, rows: int, cols: int, include_diagonal: bool = True, rigid_outer_frame: bool = True):
        """
        Build an undirected grid graph on (row x col) nodes.
        - 4-neighbor connections always included
        - if include_diagonal=True, add 4 diagonal neighbors too (8-neighborhood)
        All edges get stiffness K=101.0.
        
        Node numbering: nid(r,c) = r*col + c (row-major)
        Returns:
        A (n,n) int in {0,1}
        K (n,n) float (101.0 where A=1 else 0.0)
        """
        n = rows * cols
        A_orig = np.zeros((n, n), dtype=int)
        K_orig = np.zeros((n, n), dtype=float)

        def nid(r, c): 
            return r * cols + c
        
        def _is_outer_frame_edge(r, c, rr, cc):
            # Only orthogonal neighbors
            dr = rr - r
            dc = cc - c
            if abs(dr) + abs(dc) != 1:
                return False

            # horizontal edge on top/bottom row
            if dr == 0 and (r == 0 or r == rows - 1):
                return True

            # vertical edge on left/right col
            if dc == 0 and (c == 0 or c == cols - 1):
                return True

            return False

        # 4-neighbors
        directions = [(-1,0), (1,0), (0,-1), (0,1)]
        # if include_diagonal:
        #     directions += [(-1,-1), (-1,1), (1,-1), (1,1)]

        for r in range(rows):
            for c in range(cols):
                i = nid(r, c)
                for dr, dc in directions:
                    rr, cc = r + dr, c + dc

                    if 0 <= rr < rows and 0 <= cc < cols:
                        j = nid(rr, cc)
                        if j == i:
                            continue

                        if rigid_outer_frame and _is_outer_frame_edge(r, c, rr, cc):
                            A_orig[i, j] = A_orig[j, i] = 1
                            K_orig[i, j] = K_orig[j, i] = -1
                            continue

                        A_orig[i, j] = 1
                        K_orig[i, j] = 222.15

        # make strictly symmetric and zero diagonal
        A_orig = np.maximum(A_orig, A_orig.T)

        rigid = (K_orig < 0) | (K_orig.T < 0)
        K_sym = np.maximum(K_orig, K_orig.T)
        K_orig = np.where(rigid, -1.0, K_sym)

        np.fill_diagonal(A_orig, 0)
        np.fill_diagonal(K_orig, 0.0)
        return A_orig, K_orig
    
    def plot_fourier_viz(self, 
            F: np.ndarray, F_discr_norm: np.ndarray, A_new: np.ndarray, K_new: np.ndarray, A_orig: np.ndarray, K_orig: np.ndarray, rows: int, cols: int,
            show_cell_values: bool = True, show_node_ids: bool = False, node_dot_size: int = 35
        ):
        """
        2x2 layout:
        (0,0) 3D surface of continuous Fourier series
        (0,1) 2D imshow of continuous Fourier series
        (1,0) 2D imshow of discretized (normalized) Fourier series + values + nodes
        (1,1) same discretized plot + edges overlay (red for k=1, grey for k=0.5)

        All 2D imshow plots use origin='upper' and extent=(0,Lx,0,Ly) to stay consistent
        with matrix indexing (row 0 visually at top).
        """
        Fd = np.asarray(F_discr_norm, dtype=float)
        if Fd.ndim != 2:
            raise ValueError(f"F_discr_norm must be 2D, got shape {Fd.shape}")
        if Fd.min() < 0 or Fd.max() > 1:
            raise ValueError("F_discr_norm must already be normalized to [0,1].")

        n = rows * cols
        A_new = np.asarray(A_new)
        K_new = np.asarray(K_new, dtype=float)
        if A_new.shape != (n, n) or K_new.shape != (n, n):
            raise ValueError(f"A_new and K_new must be shape {(n,n)}, got {A_new.shape and K_new.shape}")

        if (A_orig is None) ^ (K_orig is None):
            raise ValueError("Provide BOTH A_orig and K_orig, or neither.")
        if A_orig is not None:
            A_orig = np.asarray(A_orig)
            K_orig = np.asarray(K_orig, dtype=float)
            if A_orig.shape != (n, n) or K_orig.shape != (n, n):
                raise ValueError(f"A_orig and K_orig must be shape {(n,n)}")

        Lx, Ly = self.domain_size
        ny, nx = Fd.shape

        # --- cell centers (physical coords) for discretized array ---
        x_edges = np.linspace(0.0, Lx, nx + 1)
        y_edges = np.linspace(0.0, Ly, ny + 1)
        x_centers = 0.5 * (x_edges[:-1] + x_edges[1:])
        y_centers = 0.5 * (y_edges[:-1] + y_edges[1:])
        y_centers_plot = Ly - y_centers  # origin='upper' flip

        # --- map original nodes to nearest discretized cell centers (physical coords) ---
        x_orig = np.linspace(0.0, Lx, cols)
        y_orig_plot = np.linspace(Ly, 0.0, rows)  # row 0 at top
        Xo, Yo = np.meshgrid(x_orig, y_orig_plot)

        ix = np.abs(x_centers[None, :] - Xo.ravel()[:, None]).argmin(axis=1)
        iy = np.abs(y_centers_plot[None, :] - Yo.ravel()[:, None]).argmin(axis=1)
        x_node = x_centers[ix]
        y_node = y_centers_plot[iy]

        def _draw_edges(ax, A_mat, K_mat):
            # draw edges once (i<j)
            for i in range(n):
                xi, yi = x_node[i], y_node[i]
                for j in range(i + 1, n):
                    if A_mat[i, j] == 0:
                        continue
                    kij = K_mat[i, j]
                    if kij == 222.15:
                        color, lw = "black", 1.8
                    elif kij == 195.64:
                        color, lw = "grey", 1.8
                    elif kij < 0:
                        color, lw = "red", 1.8
                    else:
                        continue
                    ax.plot([xi, x_node[j]], [yi, y_node[j]], color=color, linewidth=lw, zorder=8)

        # --- figure with 2x2, make (0,0) a 3D axis ---
        fig = plt.figure(figsize=(16, 12), constrained_layout=True)
        gs = fig.add_gridspec(2, 2)

        ax00 = fig.add_subplot(gs[0, 0], projection="3d")
        ax01 = fig.add_subplot(gs[0, 1])
        ax10 = fig.add_subplot(gs[1, 0])
        ax11 = fig.add_subplot(gs[1, 1])

        # =========================
        # (0,0) 3D surface plot
        # =========================
        # Use the same fine grid from your FourierSeries2D instance
        Xf = self.X_fine
        Yf = self.Y_fine
        ax00.plot_surface(Xf, Yf, F, linewidth=0, antialiased=True)
        ax00.set_title("Continuous Fourier Series (3D surface)")
        ax00.set_xlabel("x")
        ax00.set_ylabel("y")
        ax00.set_zlabel("F(x,y)")

        # =========================
        # (0,1) 2D continuous imshow
        # =========================
        im01 = ax01.imshow(
            F,
            origin="upper",
            extent=(0, Lx, 0, Ly),
            aspect="equal",
        )
        ax01.set_title("Continuous Fourier Series (2D)")
        ax01.set_xlabel("x")
        ax01.set_ylabel("y")
        fig.colorbar(im01, ax=ax01, fraction=0.046, pad=0.04)

        # =========================
        # (1,0) discretized + values + nodes
        # =========================
        im10 = ax10.imshow(
            Fd,
            origin="upper",
            extent=(0, Lx, 0, Ly),
            aspect="equal",
            interpolation="nearest",
            vmin=0.0, vmax=1.0,
        )
        ax10.set_title("Discretized Fourier Series (Before Optimization)")
        ax10.set_xlabel("x")
        ax10.set_ylabel("y")
        fig.colorbar(im10, ax=ax10, fraction=0.046, pad=0.04)

        if show_cell_values:
            for r_cell in range(ny):
                for c_cell in range(nx):
                    ax10.text(
                        x_centers[c_cell],
                        y_centers_plot[r_cell],
                        f"{Fd[r_cell, c_cell]:.2f}",
                        ha="center", va="center",
                        fontsize=11, zorder=5
                    )

        if A_orig is not None:
            _draw_edges(ax10, A_orig, K_orig)

        ax10.scatter(
            x_node, y_node,
            c="red", s=node_dot_size,
            edgecolors="black", linewidths=0.6,
            zorder=10
        )

        if show_node_ids:
            for rr in range(rows):
                for cc in range(cols):
                    nid = rr * cols + cc
                    k = rr * cols + cc
                    ax10.text(
                        x_node[k], y_node[k], str(nid),
                        ha="center", va="center",
                        fontsize=11, color="blue",
                        zorder=11
                    )

        # =========================
        # (1,1) discretized + edges overlay
        # =========================
        im11 = ax11.imshow(
            Fd,
            origin="upper",
            extent=(0, Lx, 0, Ly),
            aspect="equal",
            interpolation="nearest",
            vmin=0.0, vmax=1.0,
        )
        ax11.set_title("Discretized Fourier Series (After Optimization)\n(black=222.15/m, grey=195.64N/m)")
        ax11.set_xlabel("x")
        ax11.set_ylabel("y")
        fig.colorbar(im11, ax=ax11, fraction=0.046, pad=0.04)

        if show_cell_values:
            for r_cell in range(ny):
                for c_cell in range(nx):
                    ax11.text(
                        x_centers[c_cell],
                        y_centers_plot[r_cell],
                        f"{Fd[r_cell, c_cell]:.2f}",
                        ha="center", va="center",
                        fontsize=11, zorder=5
                    )

        _draw_edges(ax11, A_new, K_new)

        ax11.scatter(
            x_node, y_node,
            c="red", s=node_dot_size,
            edgecolors="black", linewidths=0.6,
            zorder=10
        )

        if show_node_ids:
            for rr in range(rows):
                for cc in range(cols):
                    nid = rr * cols + cc
                    k = rr * cols + cc
                    ax11.text(
                        x_node[k], y_node[k], str(nid),
                        ha="center", va="center",
                        fontsize=11, color="blue",
                        zorder=11
                    )

        plt.show()