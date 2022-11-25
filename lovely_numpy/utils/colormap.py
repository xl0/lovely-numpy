# AUTOGENERATED! DO NOT EDIT! File to edit: ../../nbs/03a_utils.colormap.ipynb.

# %% auto 0
__all__ = ['InfCmap']

# %% ../../nbs/03a_utils.colormap.ipynb 4
from typing import Optional
import numpy as np
import matplotlib as mpl, matplotlib.cm as cm
from matplotlib.colors import Colormap, to_rgba

from .. import lovely
from ..repr_rgb import rgb

# %% ../../nbs/03a_utils.colormap.ipynb 5
def get_cmap(cmap: str) -> Colormap:
    # Matplotlib changed the colormap interface in version 3.6, and immediately
    # marked the old one as deprecated with a warning. I want to suppot
    # both for the time being, and avoid the warning for people using 3.6+.
    major, minor, *rest = mpl.__version__.split(".")
    assert int(major) == 3 # Drop this compat code when mpl is at 4.0

    if int(minor) <= 5:
        return cm.get_cmap(cmap)
    else:
        return mpl.colormaps[cmap]


# %% ../../nbs/03a_utils.colormap.ipynb 17
class InfCmap():
    """
    Matplotlib colormap extended to have colors for +/-inf

    Parameters extept `cmap` are matplotlib color strings.
    """
    def __init__(self,
                 cmap:  Colormap, # Base matplotlib colormap
                 below: Optional[str] =None, # Values below 0
                 above: Optional[str] =None, # Values above 1
                 nan:   Optional[str] =None, # NaNs
                 ninf:  Optional[str] =None, # -inf
                 pinf:  Optional[str] =None, # +inf
                ):
        _ = cmap(0) # one call to make sure the cmap is initialized
        assert len(cmap._lut) == 259, "The colormap LUT should have 259 inputs"
        lut = cmap._lut.copy()
        
        if below: lut[256] = np.array(to_rgba(below))
        if above: lut[257] = np.array(to_rgba(above))
        if nan: lut[258] = np.array(to_rgba(nan))

        # For +/- inf, use above/below as defaults.
        tensor_cmap_ninf = np.array(to_rgba(ninf)) if ninf else lut[256]
        tensor_cmap_pinf = np.array(to_rgba(pinf)) if pinf else lut[257]

        # Remove the alpha channel, it causes probems in pad_frame_gutters().
        self.lut = np.concatenate([ lut, tensor_cmap_ninf[None], tensor_cmap_pinf[None] ])[:,:3]

    def __call__(self, t: np.ndarray):
        lut_idxs = (t*255).astype(np.uint8).astype(np.int64)
        
        lut_idxs[ t < 0. ] = 256
        lut_idxs[ t > 1. ] = 257
        lut_idxs[ np.isnan(t)] = 258

        lut_idxs[ np.isneginf(t) ] = 259
        lut_idxs[ np.isposinf(t) ] = 260
        
        return self.lut.take(lut_idxs, axis=0) # RGB added as color-last.
        
         
