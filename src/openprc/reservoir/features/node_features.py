from openprc.reservoir.features.base import BaseFeature

class NodePositions(BaseFeature):
    def __init__(self, node_ids="all"):
        self.node_ids = node_ids
    
    def transform(self, state_loader):
        return state_loader.get_node_positions(self.node_ids)