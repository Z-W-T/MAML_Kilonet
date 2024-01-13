from dataloaders import *
from renderer import *


a = CombustionDataset('hr', 100)
a.read_volume_data()
a.normalize()
vtk_draw_single_volume(a)