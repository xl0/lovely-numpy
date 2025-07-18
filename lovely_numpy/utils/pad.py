# AUTOGENERATED! DO NOT EDIT! File to edit: ../../nbs/03b_utils.pad.ipynb.

# %% auto 0
__all__ = ['pad_frame', 'pad_frame_gutters']

# %% ../../nbs/03b_utils.pad.ipynb
import numpy as np

# %% ../../nbs/03b_utils.pad.ipynb
def pad_frame(t: np.ndarray, # torch.Tensor,  # 3D+ image tensor, [...,H,W,C]
              frame_px: int=1,       # Number of pixels to pad each side.
              val :float=0):           # Value to pad with.
    """Pad H and W dimensitons of an image tensor with `val` of thickness `frame_px`"""
    assert t.ndim >= 3

    return np.pad(t, pad_width=[(0,0)] * (t.ndim-3) # Don't pad higher dims
                  + [(frame_px,frame_px)] * 2 #  H and W get padded
                  + [(0,0)], # C does not get padded
                  constant_values=val)


# %% ../../nbs/03b_utils.pad.ipynb
def pad_frame_gutters(t: np.ndarray,  # 3D+ Tensor image tensor, [...,H,W,C]
                      gutter_px=3,      # Write gutter in pixels.
                      frame_px=1):      # Black frame, in pixels
    """Add a black frame and white gutters around an image"""
    assert t.ndim >= 3
    xy_shape = t.shape[-2:]
    # gutter_px = ceil(max(xy_shape)*gutter_frac//2)
    
    # XXX This does not work for RGBA images, as the alpha channel is set to 0!
    t = pad_frame(t, frame_px=frame_px, val=0) # Black frame
    return pad_frame(t, frame_px=gutter_px, val=1) # White gutters between images
