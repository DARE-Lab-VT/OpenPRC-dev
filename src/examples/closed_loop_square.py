"""
Closed-loop Square geometry
===========================
Generates Closed-loop Square Kirigami geometry with nodes and bars.
"""

import numpy as np


class NodeList:
    """
    Custom list class that supports both flat indexing and 2D indexing [loop, angle].
    """
    def __init__(self, num_loops, num_node_per_loop):
        self.num_loops = num_loops
        self.num_node_per_loop = num_node_per_loop
        self._nodes = []
    
    def append(self, node):
        """Add a node to the list."""
        self._nodes.append(node)
    
    def __getitem__(self, key):
        """
        Support both indexing styles:
        - nodes[i] -> get node by flat index
        - nodes[loop, angle] -> get node by loop and angle
        """
        if isinstance(key, tuple):
            # 2D indexing: nodes[loop, angle]
            loop, angle = key
            if not (0 <= loop < self.num_loops):
                raise IndexError(f"Loop index {loop} out of range [0, {self.num_loops})")
            if not (0 <= angle < self.num_node_per_loop):
                raise IndexError(f"Angle index {angle} out of range [0, {self.num_node_per_loop})")

            flat_index = loop * self.num_node_per_loop + angle
            return self._nodes[flat_index]
        else:
            # Flat indexing: nodes[i]
            return self._nodes[key]
    
    def __setitem__(self, key, value):
        """Support setting nodes with both indexing styles."""
        if isinstance(key, tuple):
            loop, angle = key
            flat_index = loop * self.num_node_per_loop + angle
            self._nodes[flat_index] = value
        else:
            self._nodes[key] = value
    
    def __len__(self):
        """Return total number of nodes."""
        return len(self._nodes)
    
    def __iter__(self):
        """Allow iteration over all nodes."""
        return iter(self._nodes)
    
    def __repr__(self):
        return f"NodeList({len(self._nodes)} nodes, {self.num_loops} loops × {self.num_node_per_loop} nodes/loop)"


class ClosedLoopSquare:
    """
    Main geometry class for concentric square loop design.
    Contains inner classes for Node and Bar components.
    """
    
    class Node:
        """Inner class representing a node in the geometry."""
        
        def __init__(self, index, loop, angle, position, mass, boundary_condition=None):
            """
            Initialize a node.
            
            Args:
                index: Unique node identifier
                loop: Loop number (0 to num_loops-1)
                angle: Angular position [0-7] * 45 degrees
                position: [x, y, z] coordinates as numpy array
                mass: Node mass
                boundary_condition: Optional boundary condition (e.g., 'fixed', 'free')
            """
            self.index = index
            self.loop = loop
            self.angle = angle
            self.pos = np.array(position, dtype=float)
            self.mass = mass
            self.boundary_condition = boundary_condition or 'free'
        
        def __repr__(self):
            return f"Node(idx={self.index}, loop={self.loop}, angle={self.angle})"
    
    class Bar:
        """Inner class representing a bar connecting two nodes."""
        
        def __init__(self, index, node_a_idx, node_b_idx, stiffness, rest_length, damping):
            """
            Initialize a bar.
            
            Args:
                index: Unique bar identifier
                node_a_idx: Index of first node
                node_b_idx: Index of second node
                stiffness: Bar stiffness coefficient
                rest_length: Rest length of the bar
                damping: Damping coefficient
            """
            self.index = index
            self.node_a_idx = node_a_idx
            self.node_b_idx = node_b_idx
            self.stiffness = stiffness
            self.rest_length = rest_length
            self.damping = damping
        
        def __repr__(self):
            return f"Bar(idx={self.index}, nodes={self.node_a_idx}-{self.node_b_idx})"

    class Hinge:
        """Inner class representing a hinge connecting four nodes."""

        def __init__(self, index, node_a_idx, node_b_idx, node_c_idx, node_d_idx,
                     stiffness, rest_angle=0.0):
            """
            Initialize a hinge.

            Args:
                index: Unique hinge identifier
                node_a_idx: Index of node a (next loop)
                node_b_idx: Index of node b (current loop)
                node_c_idx: Index of node c (varies by loop parity)
                node_d_idx: Index of node d (varies by loop parity)
                stiffness: Hinge stiffness coefficient
                rest_angle: Rest angle of the hinge (default: 0.0)
            """
            self.index = index
            self.node_a_idx = node_a_idx
            self.node_b_idx = node_b_idx
            self.node_c_idx = node_c_idx
            self.node_d_idx = node_d_idx
            self.stiffness = stiffness
            self.rest_angle = rest_angle

        def __repr__(self):
            return (f"Hinge(idx={self.index}, nodes={self.node_a_idx}-"
                    f"{self.node_b_idx}-{self.node_c_idx}-{self.node_d_idx})")

    def __init__(self, num_loops, side, width, joint_width, cut_width, mass, 
                 E, thickness, damping):
        """
        Initialize the closed-loop square geometry.
        
        Args:
            num_loops: Number of concentric square loops
            side: Side length of the outermost square
            width: Width of the material (for bars within loops)
            joint_width: Width of joints (for bars between loops)
            cut_width: Width of cuts
            mass: Total mass to distribute among nodes
            E: Young's modulus for stiffness calculation
            thickness: Thickness of the material
            damping: Damping coefficient for all bars
        """
        # Geometry parameters
        self.num_loops = num_loops
        self.side = side
        self.width = width
        self.joint_width = joint_width
        self.cut_width = cut_width
        self.total_mass = mass
        self.num_node_per_loop = 4

        # Material properties
        self.E = E  # Young's modulus
        self.thickness = thickness
        self.damping = damping
        
        # Nodal direction vectors (8 positions around square at 45° intervals)
        # XY plane
        self.nodal_vectors_EVEN = np.array([
            # [1, 0, 0],      # 0°   - Right
            [1, 1, 0],      # 45°  - Top-right corner
            # [0, 1, 0],      # 90°  - Top
            [-1, 1, 0],     # 135° - Top-left corner
            # [-1, 0, 0],     # 180° - Left
            [-1, -1, 0],    # 225° - Bottom-left corner
            # [0, -1, 0],     # 270° - Bottom
            [1, -1, 0]      # 315° - Bottom-right corner
        ])

        self.nodal_vectors_ODD = np.array([
            [1, 0, 0],  # 0°   - Right
            # [1, 1, 0],  # 45°  - Top-right corner
            [0, 1, 0],  # 90°  - Top
            # [-1, 1, 0],  # 135° - Top-left corner
            [-1, 0, 0],  # 180° - Left
            # [-1, -1, 0],  # 225° - Bottom-left corner
            [0, -1, 0],  # 270° - Bottom
            # [1, -1, 0]  # 315° - Bottom-right corner
        ])

        
        # Calculate ligament lengths for each loop to calculate lumped mass of nodes
        self.ligament_lengths = np.array([
            (self.side - self.width) / 2.0 - (self.cut_width + self.width) * i
            for i in range(self.num_loops)
        ])

        # Calculate loop side lengths for each loop to calculate the position of nodes
        self.loop_radius = np.array([
            (self.side/2 - i * self.width + (0.5 - i) * self.cut_width)
            for i in range(self.num_loops)
        ])
        
        # Storage for nodes, bars, and hinges
        self.nodes = NodeList(num_loops, self.num_node_per_loop)
        self.bars = []
        self.hinges = []
        
        # Build the geometry
        self._generate_nodes()
        print(self.nodes)
        self._generate_bars()

    def _generate_nodes(self):
        """Generate all nodes for all loops."""
        node_idx = 0
        total_ligament_length = np.sum(self.ligament_lengths)
        
        for loop in range(self.num_loops):
            # Mass for this loop proportional to its ligament length
            loop_mass = self.total_mass * (self.ligament_lengths[loop] / total_ligament_length)
            node_mass = loop_mass / self.num_node_per_loop

            if loop % 2 == 0:  # Loop 0, 2, 4...
                create_node_vector = self.nodal_vectors_EVEN
            else:  # Loop 1, 3, 5...
                create_node_vector = self.nodal_vectors_ODD

            for n in range(self.num_node_per_loop):
                # Calculate position
                position = create_node_vector[n] * self.loop_radius[loop]
                
                # Determine boundary condition
                # Fix corner nodes (angles 1, 3, 5, 7) of the first loop (outermost)
                if loop == 0:
                    bc = 'fixed'
                else:
                    bc = 'free'

                # Create node
                node = self.Node(
                    index=node_idx,
                    loop=loop,
                    angle=n,
                    position=position,
                    mass=node_mass,
                    boundary_condition=bc
                )
                
                self.nodes.append(node)
                print(f'node[{node_idx}] = loop{loop}, angle = {n}')
                print(f'node[{self.nodes[loop, n]}] idx = {self.nodes[node_idx].loop}, {self.nodes[node_idx].angle}')
                node_idx += 1

    def _generate_bars(self):
        """
        Generate all bars connecting nodes within and between loops.
        
        Connectivity rules:
        - Within each loop: each node connects to the next, forming a closed loop
        - Between adjacent loops:
          * If current loop is ODD (0, 2, 4...): connect EVEN angles (0, 2, 4, 6)
          * If current loop is EVEN (1, 3, 5...): connect ODD angles (1, 3, 5, 7)
        """
        bar_idx = 0

        for loop in range(self.num_loops):

            if loop%2 == 0:
                idx_nodes_to_connect_to = [0, 1]
            else:
                idx_nodes_to_connect_to = [0, -1]

            next_loop_flag = 1
            if loop == self.num_loops - 1:
                next_loop_flag = 0
                idx_nodes_to_connect_to = [1]

            for n in range(self.num_node_per_loop):
                for i in idx_nodes_to_connect_to:
                    node_a_idx = self.nodes[loop, n].index
                    node_b_idx = self.nodes[loop + next_loop_flag, (n + i + self.num_node_per_loop) % self.num_node_per_loop].index

                    # Get node positions
                    node_a = self.nodes[node_a_idx]
                    node_b = self.nodes[node_b_idx]

                    # Calculate rest length
                    rest_length = np.linalg.norm(node_b.pos - node_a.pos)

                    # Calculate stiffness: k = E * A / L = E * thickness * joint_width / rest_length
                    # For inter-loop bars, use self.joint_width
                    cross_section_area = self.thickness * self.joint_width
                    stiffness = self.E * cross_section_area / rest_length

                    # Create bar
                    bar = self.Bar(
                        index=bar_idx,
                        node_a_idx=node_a_idx,
                        node_b_idx=node_b_idx,
                        stiffness=stiffness,
                        rest_length=rest_length,
                        damping=self.damping
                    )

                    self.bars.append(bar)
                    print(bar, bar.rest_length)
                    bar_idx += 1

    def _generate_hinges(self):
        """
        Generate all hinges connecting nodes between loops.

        Hinge rules:
        - 4 hinges per loop (at connection points between loops)
        - If loop is EVEN (0, 2, 4...):
            * Hinge nodes: [loop+1, angle], [loop, angle], [loop, angle+1], [loop, angle-1]
            * At angles 0, 2, 4, 6
        - If loop is ODD (1, 3, 5...):
            * Hinge nodes: [loop, angle], [loop-1, angle], [loop, angle+1], [loop, angle-1]
            * At angles 0, 2, 4, 6

        Stiffness calculation:
        k_hinge = E * (1/12) * width * thickness^3 / initial distance(node_c, node_d)^2
        """
        hinge_idx = 0

        # Generate hinges for each loop

        # Define the 4 hinges for each loop
        hinge_configs = [
            (0, 0, 1, 7),  # angle 0
            (2, 2, 3, 1),  # angle 2
            (4, 4, 5, 3),  # angle 4
            (6, 6, 7, 5),  # angle 6
        ]

        for loop in range(self.num_loops):
            for angle_a, angle_b, angle_c, angle_d in hinge_configs:
                # Node indices
                if loop % 2 == 0:
                    node_a = self.nodes[loop+1, angle_a]
                    node_b= self.nodes[loop, angle_b]
                else:
                    node_a = self.nodes[loop, angle_a]
                    node_b = self.nodes[loop-1, angle_b]
                node_c = self.nodes[loop, angle_c]
                node_d = self.nodes[loop, angle_d]

                # Calculate distance between node c and node d
                distance_cd = np.linalg.norm(node_c.pos - node_d.pos)

                # Calculate hinge stiffness
                # stiffness = self.E * (1.0 / 12.0) * self.width * (self.thickness ** 3) / (distance_cd ** 2) *1000
                stiffness = 10.0

                # Create hinge
                hinge = self.Hinge(
                    index=hinge_idx,
                    node_a_idx=node_a.index,
                    node_b_idx=node_b.index,
                    node_c_idx=node_c.index,
                    node_d_idx=node_d.index,
                    stiffness=stiffness,
                    rest_angle=np.pi
                )

                self.hinges.append(hinge)
                hinge_idx += 1

        # hinge_configs = [
        #     (3, 7, 1, 5),
        #     (1, 5, 7, 3)
        # ]
        #
        # for loop in range(self.num_loops):
        #     for angle_a, angle_b, angle_c, angle_d in hinge_configs:
        #         node_a = self.nodes[loop, angle_a]
        #         node_b = self.nodes[loop, angle_b]
        #         node_c = self.nodes[loop, angle_c]
        #         node_d = self.nodes[loop, angle_d]
        #
        #         # Calculate distance between node c and node d
        #         distance_cd = np.linalg.norm(node_c.pos - node_d.pos)
        #
        #         # Calculate hinge stiffness
        #         stiffness = self.E * (1.0 / 12.0) * self.width * (self.thickness ** 3) / (distance_cd ** 2)
        #
        #         # Create hinge
        #         hinge = self.Hinge(
        #             index=hinge_idx,
        #             node_a_idx=node_a.index,
        #             node_b_idx=node_b.index,
        #             node_c_idx=node_c.index,
        #             node_d_idx=node_d.index,
        #             stiffness=stiffness,
        #             rest_angle=np.pi
        #         )
        #
        #         self.hinges.append(hinge)
        #         hinge_idx += 1

        # hinge_configs = [
        #     (7, 1, 0, 0),
        #     (1, 3, 2, 2),
        #     (3, 5, 4, 4),
        #     (5, 7, 4, 4)
        # ]
        #
        # for loop in range(self.num_loops-1):
        #     for angle_a, angle_b, angle_c, angle_d in hinge_configs:
        #         node_a = self.nodes[loop, angle_a]
        #         node_b = self.nodes[loop, angle_b]
        #         node_c = self.nodes[loop+1, angle_c]
        #         node_d = self.nodes[loop, angle_d]
        #
        #         # Calculate distance between node c and node d
        #         distance_cd = np.linalg.norm(node_c.pos - node_d.pos)
        #
        #         # Calculate hinge stiffness
        #         stiffness = self.E * (1.0 / 12.0) * self.width * (self.thickness ** 3) / (distance_cd ** 2)
        #
        #         # Create hinge
        #         hinge = self.Hinge(
        #             index=hinge_idx,
        #             node_a_idx=node_a.index,
        #             node_b_idx=node_b.index,
        #             node_c_idx=node_c.index,
        #             node_d_idx=node_d.index,
        #             stiffness=stiffness,
        #             rest_angle=np.pi
        #         )
        #
        #         self.hinges.append(hinge)
        #         hinge_idx += 1

    def get_node(self, index):
        """Get node by index."""
        return self.nodes[index] if 0 <= index < len(self.nodes) else None
    
    def get_bar(self, index):
        """Get bar by index."""
        return self.bars[index] if 0 <= index < len(self.bars) else None

    def get_hinge(self, index):
        """Get hinge by index."""
        return self.hinges[index] if 0 <= index < len(self.hinges) else None

    def get_nodes_in_loop(self, loop):
        """Get all nodes in a specific loop."""
        return [node for node in self.nodes if node.loop == loop]

    def get_hinges_in_loop(self, loop):
        """Get all hinges connecting a specific loop to the next loop."""
        hinges = []
        for hinge in self.hinges:
            node_b = self.nodes[hinge.node_b_idx]
            if node_b.loop == loop:
                hinges.append(hinge)
        return hinges

    def get_connectivity_matrix(self):
        """
        Return connectivity matrix showing which nodes are connected.
        
        Returns:
            numpy array of shape (num_nodes, num_nodes) with 1 for connected nodes
        """
        num_nodes = len(self.nodes)
        connectivity = np.zeros((num_nodes, num_nodes), dtype=int)
        
        for bar in self.bars:
            connectivity[bar.node_a_idx, bar.node_b_idx] = 1
            connectivity[bar.node_b_idx, bar.node_a_idx] = 1
        
        return connectivity

    def summary(self):
        """Print summary of the geometry."""
        print(f"Closed-Loop Square Geometry")
        print(f"{'='*50}")
        print(f"Number of loops: {self.num_loops}")
        print(f"Nodes per loop: {self.num_node_per_loop}")
        print(f"Total nodes: {len(self.nodes)}")
        print(f"Total bars: {len(self.bars)}")
        print(f"Total hinges: {len(self.hinges)}")
        print(f"Total mass: {self.total_mass:.3f}")
        print(f"Ligament lengths: {self.ligament_lengths}")
        print(f"Young's modulus (E): {self.E}")
        print(f"Thickness: {self.thickness}")
        print(f"Damping: {self.damping}")
