from openprc.reservoir.features.base import BaseFeature


class BarStrains(BaseFeature):
    """Reservoir state = bar engineering strain ε = (L-L₀)/L₀ per frame."""
    def __init__(self, bar_ids="all"):
        self.bar_ids = bar_ids

    def transform(self, state_loader):
        return state_loader.get_bar_strains(self.bar_ids)


class BarStresses(BaseFeature):
    """Reservoir state = bar axial stress σ = k·ε per frame."""
    def __init__(self, bar_ids="all"):
        self.bar_ids = bar_ids

    def transform(self, state_loader):
        return state_loader.get_bar_stresses(self.bar_ids)


class BarLengths(BaseFeature):
    """Reservoir state = absolute bar length per frame (if stored)."""
    def __init__(self, bar_ids="all"):
        self.bar_ids = bar_ids

    def transform(self, state_loader):
        return state_loader.get_bar_lengths(self.bar_ids)


class BarExtensions(BaseFeature):
    """Reservoir state = bar strain (proxy for extension, uses stored strain)."""
    def __init__(self, bar_ids="all"):
        self.bar_ids = bar_ids

    def transform(self, state_loader):
        return state_loader.get_bar_extensions(self.bar_ids)