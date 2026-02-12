import numpy as np
from collections import defaultdict


def general_transform_matrix(psi, gamma, d):
    """Generate transformation matrix for Yoshimura unit."""
    return np.array([
        [np.cos(psi) ** 2 + np.cos(gamma) * np.sin(psi) ** 2,
         (1 - np.cos(gamma)) * np.sin(psi) * np.cos(psi),
         np.sin(gamma) * np.sin(psi),
         d * np.sin(psi) * np.sin(gamma / 2)],
        [(1 - np.cos(gamma)) * np.sin(psi) * np.cos(psi),
         np.sin(psi) ** 2 + np.cos(gamma) * np.cos(psi) ** 2,
         -np.sin(gamma) * np.cos(psi),
         -d * np.cos(psi) * np.sin(gamma / 2)],
        [-np.sin(gamma) * np.sin(psi),
         np.sin(gamma) * np.cos(psi),
         np.cos(gamma),
         d * np.cos(gamma / 2)],
        [0, 0, 0, 1]
    ])


def find_circumcenter(A, B, C, D):
    """Find circumcenter of quadrilateral ABCD."""
    AC = (A + C) / 2
    BD = (B + D) / 2

    a = np.linalg.norm(A - B)
    b = np.linalg.norm(B - D)
    c = np.linalg.norm(D - C)
    d = np.linalg.norm(C - A)

    s = (a + b + c + d) / 2
    R = 0.25 * ((a * b + c * d) * (a * c + b * d) * (a * d + b * c) /
                ((s - a) * (s - b) * (s - c) * (s - d))) ** 0.5

    m = (BD - AC) / np.linalg.norm(BD - AC)
    circ_cent = AC + m * (R ** 2 - (d / 2) ** 2) ** 0.5

    return circ_cent


def generate_yoshimura_geometry(n, beta, d=None, gamma=0.0, psi=0.0):
    w = 0.5 * np.tan(beta)
    l = 0.5
    b = 0.5 / np.cos(beta)

    if d is None:
        d = (np.tan(beta) ** 2 - np.tan(np.pi / (2 * n)) ** 2) ** 0.5

    params = [n, beta, d, gamma, psi]

    if d >= np.tan(beta):
        r = l / (2 * np.sin(np.pi / (2 * n)))
        base_nodes = np.array([
            [r * np.sin(np.pi / n * i),
             -r * np.cos(np.pi / n * i),
             0]
            for i in range(2 * n)
        ])
        mid_nodes = np.array([
            [r * np.sin(np.pi / n * i),
             -r * np.cos(np.pi / n * i),
             d / 2]
            for i in range(2 * n)
        ])

        top_nodes = np.array([
            [r * np.sin(np.pi / n * i),
             -r * np.cos(np.pi / n * i),
             d]
            for i in range(2 * n)
        ])
    else:
        base_nodes = np.zeros((2 * n, 3))
        mid_nodes = np.zeros((2 * n, 3))
        top_nodes = np.zeros((2 * n, 3))

        r = 1 / (2 * np.sin(np.pi / n))
        # Generate base polygon vertices
        base = np.array([
            [r * np.sin(2 * np.pi / n * i),
             -r * np.cos(2 * np.pi / n * i),
             0, 1]
            for i in range(n)
        ]).T

        # Transform to get top polygon
        T = general_transform_matrix(psi, gamma, d)
        top = T @ base

        # Calculate midpoint positions
        mid = np.zeros((4, 2 * n))
        centers = np.zeros((2 * n, 3))

        for i in range(n):
            A = base[:3, i]
            B = base[:3, (i + 1) % n]
            C = top[:3, i]
            D = top[:3, (i + 1) % n]

            p = np.cross(C - B, D - A)
            s = np.linalg.norm((A + B) / 2 - (C + D) / 2)
            a = 1 / (2 * np.cos(beta))

            x_ = w ** 2 - (s / 2) ** 2 + 1e-20
            x = np.abs(x_) ** 0.5 if x_ >= -1e-2 else 0.0

            if np.linalg.norm(p) < 1e-10:
                # Degenerate case - planar quadrilateral
                A_ = base[:3, (i + n // 2) % n]
                B_ = base[:3, (i + 1 + n // 2) % n]
                C_ = top[:3, (i + n // 2) % n]
                D_ = top[:3, (i + 1 + n // 2) % n]

                ct = (A + B + C + D) / 4

                if n % 2 == 0:
                    q_hat = ct - (A_ + B_ + C_ + D_) / 4
                else:
                    q_hat = ct - (B_ + D_) / 2

                centers[i] = ct + x * q_hat / np.linalg.norm(q_hat)
            else:
                ct = find_circumcenter(A, B, C, D)
                centers[i] = ct - p / np.linalg.norm(p) * np.abs(a ** 2 - np.linalg.norm(A - ct) ** 2) ** 0.5

        # Assign mid-edge vertices
        for i in range(n):
            B = base[:3, (i + 1) % n]
            D = top[:3, (i + 1) % n]

            mid[:3, 2 * i] = centers[i]

            if np.linalg.norm(B - D) < 1.95 * w:
                mid[:3, 2 * i + 1] = (centers[i] + centers[(i + 1) % n]) / 2
            else:
                mid[:3, 2 * i + 1] = (B + D) / 2

            mid[3, :] = 1.0

        for i in range(n):
            base_nodes[2 * i] = base[:3, i]
            base_nodes[2 * i + 1] = (base[:3, i] + base[:3, (i + 1) % n]) / 2
            mid_nodes[2 * i] = mid[:3, 2 * i - 1]
            mid_nodes[2 * i + 1] = mid[:3, 2 * i]
            top_nodes[2 * i] = top[:3, i]
            top_nodes[2 * i + 1] = (top[:3, i] + top[:3, (i + 1) % n]) / 2

    nodes = []
    nodes.extend(base_nodes)  # indices 0 to 2n-1
    nodes.extend(mid_nodes)  # indices 2n to 4n-1
    nodes.extend(top_nodes)  # indices 4n to 6n-1
    nodes = np.array(nodes)

    nodes = nodes + np.random.rand(len(nodes), 3) * 0.0

    # Index helper functions for the new structure
    def base_idx(i):
        """Get index in base layer (0 to 2n-1)"""
        return i % (2 * n)

    def mid_idx(i):
        """Get index in mid layer (2n to 4n-1)"""
        return 2 * n + i % (2 * n)

    def top_idx(i):
        """Get index in top layer (4n to 6n-1)"""
        return 4 * n + i % (2 * n)

    bars = []
    hinges = []
    faces = []

    for i in range(2 * n):
        j = i + 1
        k = i - 1

        bars.append((base_idx(i), base_idx(j), l))
        bars.append((mid_idx(i), mid_idx(j), l))
        bars.append((top_idx(i), top_idx(j), l))
        bars.append((base_idx(i), mid_idx(i), w))
        bars.append((mid_idx(i), top_idx(i), w))

        if i % 2 == 0:
            bars.append((base_idx(i), mid_idx(j), b))
            bars.append((base_idx(i), mid_idx(k), b))
            bars.append((top_idx(i), mid_idx(j), b))
            bars.append((top_idx(i), mid_idx(k), b))

            faces.append((base_idx(i), mid_idx(i), mid_idx(j)))
            faces.append((base_idx(i), mid_idx(i), mid_idx(k)))
            faces.append((base_idx(i), base_idx(j), mid_idx(j)))
            faces.append((base_idx(i), base_idx(k), mid_idx(k)))
            faces.append((top_idx(i), mid_idx(i), mid_idx(j)))
            faces.append((top_idx(i), mid_idx(i), mid_idx(k)))
            faces.append((top_idx(i), top_idx(j), mid_idx(j)))
            faces.append((top_idx(i), top_idx(k), mid_idx(k)))

            hinges.append((base_idx(i), mid_idx(j), base_idx(j), mid_idx(i), np.pi, 'fold'))
            hinges.append((top_idx(i), mid_idx(j), top_idx(j), mid_idx(i), np.pi, 'fold'))
            hinges.append((mid_idx(i), mid_idx(j), top_idx(i), base_idx(i), np.pi, 'fold'))
            hinges.append((mid_idx(i), mid_idx(k), top_idx(i), base_idx(i), np.pi, 'fold'))
            hinges.append((mid_idx(i), base_idx(k), base_idx(i), mid_idx(k), np.pi, 'fold'))
            hinges.append((mid_idx(i), top_idx(k), top_idx(i), mid_idx(k), np.pi, 'fold'))
            hinges.append((top_idx(i), mid_idx(i), mid_idx(j), mid_idx(k), np.pi, 'facet'))
            hinges.append((base_idx(i), mid_idx(i), mid_idx(j), mid_idx(k), np.pi, 'facet'))
        else:
            hinges.append((base_idx(i), mid_idx(i), base_idx(j), base_idx(k), -np.pi, 'facet'))
            hinges.append((top_idx(i), mid_idx(i), top_idx(j), top_idx(k), -np.pi, 'facet'))

    return nodes, bars, hinges, faces, params
