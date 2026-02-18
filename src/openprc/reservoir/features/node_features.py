import numpy as np
from openprc.reservoir.features.base import BaseFeature

class NodePositions(BaseFeature):
    def __init__(self, node_ids="all", dims="all"):
        self.node_ids = node_ids
        self.dims = dims
    
    def transform(self, state_loader):
        return state_loader.get_node_positions(self.node_ids, self.dims)

    def get_feature_info(self, state_loader):
        if self.node_ids == "all":
            num_nodes = state_loader.get_node_positions(reshape_output=False).shape[1]
            node_ids_to_use = list(range(num_nodes))
        else:
            node_ids_to_use = list(self.node_ids)
        
        if self.dims == "all":
            num_dims = state_loader.get_node_positions(node_ids=[0], reshape_output=False).shape[2]
            dims_to_use = list(range(num_dims))
        else:
            dims_to_use = list(self.dims)

        feature_info = []
        for node_id in node_ids_to_use:
            for dim in dims_to_use:
                feature_info.append({'node_id': node_id, 'dim': dim, 'type': 'NodePosition'})
        return feature_info


class NodeDisplacements(BaseFeature):
    def __init__(self, reference_node, node_ids="all", dims="all"):
        if not isinstance(reference_node, int):
            raise TypeError("reference_node must be an integer ID.")
        self.reference_node = reference_node
        self.node_ids = node_ids
        self.dims = dims

    def transform(self, state_loader):
        # Get the 3D position data for all nodes
        all_positions = state_loader.get_node_positions(node_ids="all", dims="all", reshape_output=False)
        
        # Extract the reference node's position for broadcasting
        # Shape: [T, 1, Dims]
        ref_pos = all_positions[:, [self.reference_node], :]
        
        # Determine which node IDs to use for displacement calculation.
        # The reference node itself is excluded, as its displacement would be zero.
        if self.node_ids == "all":
            num_nodes = all_positions.shape[1]
            node_ids_to_use = [i for i in range(num_nodes) if i != self.reference_node]
        else:
            # If user provides specific node_ids, filter out the reference_node if present
            node_ids_to_use = [i for i in list(self.node_ids) if i != self.reference_node]

        # Select the positions of the target nodes
        target_positions = all_positions[:, node_ids_to_use, :]

        # Calculate displacements relative to the reference node
        relative_displacements = target_positions - ref_pos
        
        # Select the specified dimensions (DoFs)
        if self.dims != "all":
            dims_list = list(self.dims)
            relative_displacements = relative_displacements[:, :, dims_list]

        # Reshape to the final [T, Features] format
        total_frames = relative_displacements.shape[0]
        return relative_displacements.reshape(total_frames, -1)

    def get_feature_info(self, state_loader):
        if self.node_ids == "all":
            num_nodes = state_loader.get_node_positions(reshape_output=False).shape[1]
            node_ids_to_use = [i for i in range(num_nodes) if i != self.reference_node]
        else:
            node_ids_to_use = [i for i in list(self.node_ids) if i != self.reference_node]
        
        if self.dims == "all":
            num_dims = state_loader.get_node_positions(node_ids=[0], reshape_output=False).shape[2]
            dims_to_use = list(range(num_dims))
        else:
            dims_to_use = list(self.dims)

        feature_info = []
        for node_id in node_ids_to_use:
            for dim in dims_to_use:
                feature_info.append({'node_id': node_id, 'dim': dim, 'type': 'NodeDisplacement'})
        return feature_info