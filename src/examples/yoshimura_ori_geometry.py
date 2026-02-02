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
    r = 1 / (2 * np.sin(np.pi / n))
    w = 0.5 * np.tan(beta)
    if d is None:
        d = (np.tan(beta) ** 2 - np.tan(np.pi / (2 * n)) ** 2) ** 0.5

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

    params = [n, beta, d, gamma, psi]

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
            # Non-planar quadrilateral
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

    # Create three sets of 2n nodes each: base_nodes, mid_nodes, top_nodes
    # Each set contains: [vertex, edge_midpoint, vertex, edge_midpoint, ...]
    # base_nodes = np.zeros((2 * n, 3))
    # mid_nodes = np.zeros((2 * n, 3))
    # top_nodes = np.zeros((2 * n, 3))

    base_nodes = np.array([
        [r * np.sin(np.pi / (n) * i),
         -r * np.cos(np.pi / n * i),
         0]
        for i in range(2 * n)
    ])

    mid_nodes = np.array([
        [r * np.sin(np.pi / n * i),
         -r * np.cos(np.pi / n * i),
         w]
        for i in range(2 * n)
    ])

    top_nodes = np.array([
        [r * np.sin(np.pi / n * i),
         -r * np.cos(np.pi / n * i),
         2 * w]
        for i in range(2 * n)
    ])

    # for i in range(n):
    #     # Base layer: vertex at 2*i, edge midpoint at 2*i+1
    #     base_nodes[2 * i] = base[:3, i]
    #     base_nodes[2 * i + 1] = (base[:3, i] + base[:3, (i + 1) % n]) / 2
    #
    #     # Mid layer: interior vertices (from centers and edge mids)
    #     mid_nodes[2 * i] = mid[:3, 2 * i - 1]
    #     mid_nodes[2 * i + 1] = mid[:3, 2 * i]
    #
    #     # Top layer: vertex at 2*i, edge midpoint at 2*i+1
    #     top_nodes[2 * i] = top[:3, i]
    #     top_nodes[2 * i + 1] = (top[:3, i] + top[:3, (i + 1) % n]) / 2

    # Assemble node list: [base_nodes, mid_nodes, top_nodes]
    nodes = []
    nodes.extend(base_nodes)  # indices 0 to 2n-1
    nodes.extend(mid_nodes)  # indices 2n to 4n-1
    nodes.extend(top_nodes)  # indices 4n to 6n-1
    nodes = np.array(nodes)

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

    # Generate bars and faces
    bars = []
    hinges = []
    faces = []

    for i in range(2 * n):
        j = i + 1
        k = i - 1

        # edges
        i1, i2 = base_idx(i), base_idx(j)
        length = np.linalg.norm(nodes[i1] - nodes[i2])
        bars.append((i1, i2, length))

        i1, i2 = mid_idx(i), mid_idx(j)
        length = np.linalg.norm(nodes[i1] - nodes[i2])
        bars.append((i1, i2, length))

        i1, i2 = top_idx(i), top_idx(j)
        length = np.linalg.norm(nodes[i1] - nodes[i2])
        bars.append((i1, i2, length))

        # base to mid
        i1, i2 = base_idx(i), mid_idx(i)
        length = np.linalg.norm(nodes[i1] - nodes[i2])
        bars.append((i1, i2, length))

        # mid to top
        i1, i2 = mid_idx(i), top_idx(i)
        length = np.linalg.norm(nodes[i1] - nodes[i2])
        bars.append((i1, i2, length))

        if i % 2 == 0:
            # mid to base
            i1, i2 = base_idx(i), mid_idx(j)
            length = np.linalg.norm(nodes[i1] - nodes[i2])
            bars.append((i1, i2, length))

            i1, i2 = base_idx(i), mid_idx(k)
            length = np.linalg.norm(nodes[i1] - nodes[i2])
            bars.append((i1, i2, length))

            # faces
            faces.append((base_idx(i), mid_idx(i), mid_idx(j)))
            faces.append((base_idx(i), mid_idx(i), mid_idx(k)))
            faces.append((base_idx(i), base_idx(j), mid_idx(j)))
            faces.append((base_idx(i), base_idx(k), mid_idx(k)))

            # mid to top
            i1, i2 = top_idx(i), mid_idx(j)
            length = np.linalg.norm(nodes[i1] - nodes[i2])
            bars.append((i1, i2, length))

            i1, i2 = top_idx(i), mid_idx(k)
            length = np.linalg.norm(nodes[i1] - nodes[i2])
            bars.append((i1, i2, length))

            # faces
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

        hinges.append((base_idx(i), mid_idx(i), base_idx(j), base_idx(k), np.pi - np.pi / n, 'facet'))
        hinges.append((top_idx(i), mid_idx(i), top_idx(j), top_idx(k), np.pi + np.pi / n, 'facet'))

    edge_faces = defaultdict(list)

    for face_idx, face in enumerate(faces):
        # Get all three edges of this triangular face
        edges = [
            tuple(sorted([face[0], face[1]])),
            tuple(sorted([face[1], face[2]])),
            tuple(sorted([face[2], face[0]]))
        ]
        for edge in edges:
            edge_faces[edge].append(face_idx)

    return nodes, bars, hinges, faces, params
