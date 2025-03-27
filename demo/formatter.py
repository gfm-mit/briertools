import numpy as np
import matplotlib.pyplot as plt
from matplotlib.scale import ScaleBase
from matplotlib.transforms import Transform
from matplotlib.ticker import FuncFormatter

# Define custom transform
class Ln1MinusXTransform(Transform):
    input_dims = 1
    output_dims = 1
    is_separable = True
    
    def transform_non_affine(self, a):
        return np.log(1 - a)
    
    def inverted(self):
        return InvertedLn1MinusXTransform()

class InvertedLn1MinusXTransform(Transform):
    input_dims = 1
    output_dims = 1
    is_separable = True
    
    def transform_non_affine(self, a):
        return 1 - np.exp(a)
    
    def inverted(self):
        return Ln1MinusXTransform()

# Define custom scale
class Ln1MinusXScale(ScaleBase):
    name = 'ln1minusx'
    
    def get_transform(self):
        return Ln1MinusXTransform()
    
    def set_default_locators_and_formatters(self, axis):
        axis.set_major_formatter(FuncFormatter(lambda x, pos: f"{x:.2f}"))

# Register the scale
import matplotlib.scale as mscale
mscale.register_scale(Ln1MinusXScale)