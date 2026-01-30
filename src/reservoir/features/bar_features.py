from .base import BaseFeature

class BarLengths(BaseFeature):
    """
    Reservoir State = Absolute length of springs.
    """
    def __init__(self, bar_ids="all"):
        self.bar_ids = bar_ids

    def transform(self, state_loader):
        # Calls the method we just wrote in Option 1
        return state_loader.get_bar_lengths(self.bar_ids)

class BarExtensions(BaseFeature):
    """
    Reservoir State = Change in spring length (Deformation).
    Physical interpretation: This is proportional to potential energy.
    """
    def __init__(self, bar_ids="all"):
        self.bar_ids = bar_ids

    def transform(self, state_loader):
        # Calls the method we just wrote in Option 2
        return state_loader.get_bar_extensions(self.bar_ids)