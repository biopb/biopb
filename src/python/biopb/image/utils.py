import numpy as np
from . import Pixels, BinData, ROI, Rectangle, Point, Mask

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


def deserialize_to_numpy(pixels:Pixels, *, singleton_t:bool=True) -> np.ndarray:
    '''  convert protobuf ImageData to a numpy array

    Args:
        pixels: protobuf data
    Keyword Args:
        singleton_t: data should have one time point
    
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
    )

    # The dimension_order describe axis order but in the F_order convention
    # Numpy default is C_order, so we reverse the sequence. We expect the 
    # final dimension order to be "ZYXC"
    dim_order_c = pixels.dimension_order[::-1].upper()
    dims = dict(
        Z = pixels.size_z or 1,
        Y = pixels.size_y or 1,
        X = pixels.size_x or 1,
        C = pixels.size_c or 1,
        T = 1,
    )

    if not singleton_t and pixels.size_t:
        dims['T'] = pixels.size_t

    dim_orig = [dim_order_c.find(k) for k in "ZYXCT"]
    shape_orig = [ dims[k] for k in dim_order_c ]

    np_img = np_img.reshape(shape_orig).transpose(dim_orig)

    if singleton_t:
        np_img = np_img.squeeze(axis=-1) # remove T

    return np_img


def roi_to_mask(roi: ROI, mask: np.ndarray) -> np.ndarray:
    mask_ = np.zeros_like(mask, dtype="uint8")
    dim = mask_.ndim
    
    assert dim == 2 or dim == 3, f'Ilegal mask dimension {dim}.'

    def _get_int_point(p):
        return (int(p.z), int(p.y), int(p.x))

    roi_type = roi.WhichOneof('shape')
    if roi_type == "point":
        if dim == 3:
            mask_[_get_int_point(roi.point)] = 1
        else:
            mask_[_get_int_point(roi.point)[1:]] = 1

    elif roi_type == "rectangle":
        tl = _get_int_point(roi.rectangle.top_left)
        br = _get_int_point(roi.rectangle.bottom_right)
        if dim == 3:
            mask_[tl[0]:br[0], tl[1]:br[1], tl[2]:br[2]] = 1
        else:
            mask_[tl[1]:br[1], tl[2]:br[2]] = 1

    elif roi_type == 'polygon':
        import cv2

        assert dim == 2, f'Unimplemented: 3d polygon'

        points = np.array([_get_int_point(p)[1:] for p in roi.polygon.points])
        points = points.reshape(-1, 1, 2)[:, :, ::-1] # reverse x, y

        cv2.fillPoly(mask_, [points], color=1)
    
    elif roi_type == 'mask':
        tl = _get_int_point(roi.mask.rectangle.top_left)
        br = _get_int_point(roi.mask.rectangle.bottom_right)

        bitorder = 'big' if roi.mask.bin_data.endianness == 0 else 'little'
        data = np.frombuffer(roi.mask.bin_data.data, dtype='uint8')
        data = np.unpackbits(data, bitorder=bitorder)

        if dim == 3:
            rect = mask_[tl[0]:br[0], tl[1]:br[1], tl[2]:br[2]]
        else:
            rect = mask_[tl[1]:br[1], tl[2]:br[2]]

        rect[:] = data[:rect.size].reshape(rect.shape)

    else:
        raise NotImplementedError(f"ROI type: {roi_type}")

    return mask_.astype(mask.dtype)


def mask_to_roi(mask: np.ndarray, *, bitoder:str='big') -> ROI:
    dim = mask.ndim
    
    if dim == 2:
        yp, xp = np.where(mask)
        ymin, xmin = yp.min(), xp.min()
        ymax, xmax = yp.max() + 1, xp.max() + 1
        rect = Rectangle(
            top_left = Point(y=ymin, x=xmin),
            bottom_right = Point(y=ymax, x=xmax),
        )
        pixels = mask[ymin:ymax, xmin:xmax]

    elif dim == 3:
        zp, yp, xp = np.where(mask)
        zmin, ymin, xmin = zp.min(), yp.min(), xp.min()
        zmax, ymax, xmax = zp.max() + 1, yp.max() + 1, xp.max() + 1
        rect = Rectangle(
            top_left = Point(z=zmin, y=ymin, x=xmin),
            bottom_right = Point(z=zmax, y=ymax, x=xmax),
        )
        pixels = mask[zmin:zmax, ymin:ymax, xmin:xmax]

    roi = ROI( mask = Mask(
        rectangle = rect,
        bin_data = BinData(
            data = np.packbits(pixels, bitorder=bitoder).tobytes(),
            endianness = 0 if bitoder == 'big' else 1,
        )),
    )
 
    return roi
