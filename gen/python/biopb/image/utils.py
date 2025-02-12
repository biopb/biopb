import numpy as np
from . import Pixels, BinData

def serialize_from_numpy(np_img: np.ndarray, **kwargs)->Pixels:
    '''  convert numpy array representation of image to protobuf representation

    Args:
        np_img: image in numpy array. The dimension order is assumed to be [Y, X] for 
            2d array, [Y, X, C] for 3d array and [Z, Y, X, C] for 4D array
        **kwargs: additional metadata, e.g. physical_size_x etc (pixel size)

    Returns:
        protobuf Pixels
    '''
    byteorder = np_img.dtype.byteorder
    if byteorder == "=":
        import sys
        byteorder = "<" if sys.byteorder == 'little' else ">"

    endianness = 1 if byteorder == "<" else 0

    if np_img.ndim == 2:
        np_img = np_img[np.newaxis, :, :, np.newaxis]
    elif np_img.ndim == 3:
        np_img = np_img[np.newaxis, :, :, :]
    elif np_img.ndim != 4:
        raise ValueError(f"Cannot intepret data of dim {np_img.ndim}.")

    return Pixels(
        bindata = BinData(data=np_img.tobytes(), endianness=endianness),
        size_x = np_img.shape[2],
        size_y = np_img.shape[1],
        size_c = np_img.shape[3],
        size_z = np_img.shape[0],
        dimension_order = "CXYZT",
        dtype = np_img.dtype.str,
        **kwargs,
    )


def deserialize_to_numpy(pixels:Pixels) -> np.ndarray:
    '''  convert protobuf ImageData to a numpy array

    Args:
        pixels: protobuf data
    
    Returns:
        4d Numpy array in [Z, Y, X, C] order. Singleton dimensions are kept as is.
        Note the np array has a fixed dimension order, independent of the input 
        stream. The dtype and byteorder of the np array is the same as the input.
    '''
    def _get_dtype(pixels:Pixels) -> np.dtype:
        dt = np.dtype(pixels.dtype)

        if pixels.bindata.endianness == BinData.Endianness.BIG:
            dt = dt.newbyteorder(">")
        else:
            dt = dt.newbyteorder("<")
        
        return dt

    if pixels.size_t > 1:
        raise ValueError("Image data has a non-singleton T dimension.")

    np_img = np.frombuffer(
        pixels.bindata.data, 
        dtype=_get_dtype(pixels),
    ).astype("float32")

    # The dimension_order describe axis order but in the F_order convention
    # Numpy default is C_order, so we reverse the sequence. Lacss expect the 
    # final dimension order to be "ZYXC"
    dim_order_c = pixels.dimension_order[::-1].upper()
    dims = dict(
        Z = pixels.size_z or 1,
        Y = pixels.size_y or 1,
        X = pixels.size_x or 1,
        C = pixels.size_c or 1,
        T = 1,
    )
    dim_orig = [dim_order_c.find(k) for k in "ZYXCT"]
    shape_orig = [ dims[k] for k in dim_order_c ]

    np_img = np_img.reshape(shape_orig).transpose(dim_orig)

    np_img = np_img.squeeze(axis=-1) # remove T

    return np_img
