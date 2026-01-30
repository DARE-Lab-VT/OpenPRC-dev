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
    """
    Generate Yoshimura origami unit geometry in folded configuration.

    Parameters
    ----------
    n : int
        Number of sides in the polygon base
    beta : float
        Sector angle parameter
    psi : float
        Rotation angle (default: 0.0)
    T0 : np.ndarray, optional
        Initial transformation matrix (4x4)

    Returns
    -------
    nodes : np.ndarray
        Array of node positions, shape (N, 3)
    bars : list of tuples
        List of (node_i, node_j, length) for each bar
    faces : list of tuples
        List of (node_i, node_j, node_k) for each triangular face
    """

    l = 0.075  # m
    # Calculate derived parameters for folded configuration
    r = 1 / (2 * np.sin(np.pi / n))
    w = 0.5 * np.tan(beta)

    # Folded configuration: gamma=0, d calculated from beta
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
    base_nodes = np.zeros((2 * n, 3))
    mid_nodes = np.zeros((2 * n, 3))
    top_nodes = np.zeros((2 * n, 3))

    for i in range(n):
        # Base layer: vertex at 2*i, edge midpoint at 2*i+1
        base_nodes[2 * i] = base[:3, i]
        base_nodes[2 * i + 1] = (base[:3, i] + base[:3, (i + 1) % n]) / 2

        # Mid layer: interior vertices (from centers and edge mids)
        mid_nodes[2 * i] = mid[:3, 2 * i - 1]
        mid_nodes[2 * i + 1] = mid[:3, 2 * i]

        # Top layer: vertex at 2*i, edge midpoint at 2*i+1
        top_nodes[2 * i] = top[:3, i]
        top_nodes[2 * i + 1] = (top[:3, i] + top[:3, (i + 1) % n]) / 2

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

        if i % 2 == 0:
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
        i1, i2 = mid_idx(i), top_idx(i)
        length = np.linalg.norm(nodes[i1] - nodes[i2])
        bars.append((i1, i2, length))

        if i % 2 == 0:
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

    def get_layer(idx):
        """Return which layer a node belongs to: 'base', 'mid', or 'top'"""
        if idx < 2 * n:
            return 'base'
        elif idx < 4 * n:
            return 'mid'
        else:
            return 'top'

    def is_vertex(idx):
        """Check if node is a vertex (even local index) or edge midpoint (odd)"""
        local_idx = idx % (2 * n)
        return local_idx % 2 == 0

    # Calculate dihedral angle between two faces sharing an edge
    def calc_dihedral(face1, face2, edge):
        """
        Calculate dihedral angle between two triangular faces sharing an edge.
        Returns angle in radians.
        """
        # Find the non-shared vertices
        v1_set = set(face1)
        v2_set = set(face2)
        edge_set = set(edge)

        wing1 = list(v1_set - edge_set)[0]
        wing2 = list(v2_set - edge_set)[0]

        # Get positions
        p_edge1 = nodes[edge[0]]
        p_edge2 = nodes[edge[1]]
        p_wing1 = nodes[wing1]
        p_wing2 = nodes[wing2]

        # Edge vector
        edge_vec = p_edge2 - p_edge1
        edge_vec = edge_vec / np.linalg.norm(edge_vec)

        # Vectors to wings
        v1 = p_wing1 - p_edge1
        v2 = p_wing2 - p_edge1

        # Project onto plane perpendicular to edge
        v1_perp = v1 - np.dot(v1, edge_vec) * edge_vec
        v2_perp = v2 - np.dot(v2, edge_vec) * edge_vec

        # Normalize
        v1_perp = v1_perp / (np.linalg.norm(v1_perp) + 1e-10)
        v2_perp = v2_perp / (np.linalg.norm(v2_perp) + 1e-10)

        # Angle between them
        cos_angle = np.clip(np.dot(v1_perp, v2_perp), -1, 1)
        angle = np.arccos(cos_angle)

        # Determine sign using cross product
        cross = np.cross(v1_perp, v2_perp)
        if np.dot(cross, edge_vec) < 0:
            angle = 2 * np.pi - angle

        return angle

    def classify_edge(edge):
        """
        Classify edge type for determining hinge stiffness.
        Returns: 'fold' for crease lines, 'facet' for face diagonals
        """
        idx1, idx2 = edge
        layer1, layer2 = get_layer(idx1), get_layer(idx2)
        is_v1, is_v2 = is_vertex(idx1), is_vertex(idx2)

        # Edges within same layer (ring edges)
        if layer1 == layer2:
            # These are the polygon edges - typically fold creases
            return 'fold'

        # Edges between layers
        # Base-to-mid or mid-to-top connections
        if (layer1 == 'base' and layer2 == 'mid') or \
                (layer1 == 'mid' and layer2 == 'base') or \
                (layer1 == 'mid' and layer2 == 'top') or \
                (layer1 == 'top' and layer2 == 'mid'):

            # Vertex-to-vertex or midpoint-to-midpoint: fold crease
            if is_v1 == is_v2:
                return 'fold'
            # Vertex-to-midpoint: typically facet diagonal
            else:
                return 'facet'

        return 'facet'  # default

    for edge, face_indices in edge_faces.items():
        if len(face_indices) != 2:
            # Boundary edge - no hinge needed
            continue

        face1 = faces[face_indices[0]]
        face2 = faces[face_indices[1]]

        # Find wing vertices (non-shared vertices)
        v1_set = set(face1)
        v2_set = set(face2)
        edge_set = set(edge)

        wing1 = list(v1_set - edge_set)[0]
        wing2 = list(v2_set - edge_set)[0]

        # Calculate rest angle from current geometry
        rest_angle = calc_dihedral(face1, face2, edge)

        # Classify edge and select stiffness
        edge_type = classify_edge(edge)
        hinges.append([edge[0], edge[1], wing1, wing2, rest_angle, edge_type])
    return nodes, bars, hinges, faces, params
