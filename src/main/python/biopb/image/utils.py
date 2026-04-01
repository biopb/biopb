import warnings

import numpy as np
from . import Pixels, BinData, ROI, Rectangle, Point, Mask


def _canonicalize_dtype(dtype_str: str) -> str:
    """Strip byteorder prefix from dtype for serialization.

    Byteorder is handled separately by BinData.endianness field, so the dtype
    string should only contain the type kind and size (e.g., 'u1', 'f4', 'i2').

    Args:
        dtype_str: Numpy dtype string (e.g., '|u1', '<f4', '>i2', '=u2')

    Returns:
        Canonical dtype string without byteorder prefix (e.g., 'u1', 'f4', 'i2')
    """
    dt = np.dtype(dtype_str)
    kind_map = {'u': 'u', 'i': 'i', 'f': 'f', 'c': 'c'}
    return kind_map[dt.kind] + str(dt.itemsize)


def serialize_from_numpy(np_img: np.ndarray, dimension_order: str = "CXYZT", **kwargs)->Pixels:
    '''  convert numpy array representation of image to protobuf representation

    Args:
        np_img: image in numpy array. The dimension order is assumed to be [Y, X] for
            2d array, [Y, X, C] for 3d array and [Z, Y, X, C] for 4D array
        dimension_order: string describing dimension order in the output protobuf.
            Default is "CXYZT" (C-order convention where first letter varies fastest).
            Other common values are "XYZCT" (X varies fastest).
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
        raise ValueError(f"Cannot interpret data of dim {np_img.ndim}.")

    return Pixels(
        bindata = BinData(data=np_img.tobytes(), endianness=endianness),
        size_x = np_img.shape[2],
        size_y = np_img.shape[1],
        size_c = np_img.shape[3],
        size_z = np_img.shape[0],
        dimension_order = dimension_order,
        dtype = _canonicalize_dtype(np_img.dtype.str),
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
    # Check for endianness conflict between dtype prefix and BinData field
    dtype_str = pixels.dtype
    if dtype_str and dtype_str[0] in '<>|=':
        dtype_prefix = dtype_str[0]
        # '=' means native byteorder, which could match either
        if dtype_prefix != '=':
            expected_endian = '<' if pixels.bindata.endianness == BinData.Endianness.LITTLE else '>'
            if dtype_prefix != expected_endian:
                endianness_name = "LITTLE" if pixels.bindata.endianness == BinData.Endianness.LITTLE else "BIG"
                warnings.warn(
                    f"Endianness conflict: dtype={dtype_str} indicates {dtype_prefix}-endian "
                    f"but BinData.endianness={endianness_name}. "
                    f"Using BinData.endianness as authoritative source."
                )

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
    """Convert a ROI protobuf to a binary mask.

    Args:
        roi: ROI protobuf message containing shape (point, rectangle, polygon, or mask).
        mask: Template numpy array defining output shape and dtype.

    Returns:
        Binary mask as numpy array with same shape/dtype as input mask.

    Raises:
        ValueError: If mask dimension is not 2 or 3.
        NotImplementedError: For unsupported ROI types or 3D polygons.
    """
    mask_ = np.zeros_like(mask, dtype="uint8")
    dim = mask_.ndim

    if dim not in (2, 3):
        raise ValueError(f'Illegal mask dimension {dim}.')

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
        if dim != 2:
            raise NotImplementedError('3D polygon not supported')

        try:
            import cv2
        except ImportError as e:
            raise ImportError(
                "cv2 (opencv-python) is required for polygon ROI conversion. "
                "Install it with: pip install opencv-python"
            ) from e

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


def mask_to_roi(mask: np.ndarray, *, bitorder: str = 'big') -> ROI:
    """Convert a binary mask to a ROI protobuf.

    Args:
        mask: Binary mask as numpy array (2D or 3D).
        bitorder: Bit order for packing ('big' or 'little'). Defaults to 'big'.

    Returns:
        ROI protobuf message containing the mask data.
    """
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
            data = np.packbits(pixels, bitorder=bitorder).tobytes(),
            endianness = 0 if bitorder == 'big' else 1,
        )),
    )
 
    return roi
