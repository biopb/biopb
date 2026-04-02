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


def serialize_from_numpy(
    np_img: np.ndarray,
    dimension_order: str = "CXYZT",
    np_index_order: str = None,
    **kwargs
) -> Pixels:
    '''Convert numpy array representation of image to protobuf representation.

    Args:
        np_img: image as numpy array (any memory order is accepted)
        dimension_order: F-order string describing dimension order in the output protobuf.
            Must be exactly 5 characters (a permutation of "XYZCT").
            First letter varies fastest in the serialized bytes.
            Default is "CXYZT".
        np_index_order: Numpy index order string describing which axis corresponds to which dimension.
            First letter corresponds to numpy axis 0, second to axis 1, etc.
            Must be 2-5 characters (a permutation of subset of "XYZCT").
            If None (default), inferred from np_img.ndim:
            - 2D -> "YX"
            - 3D -> "YXC"
            - 4D -> "ZYXC"
            - 5D -> "TZYXC"
        **kwargs: additional metadata, e.g. physical_size_x etc (pixel size)

    Returns:
        protobuf Pixels
    '''
    # Validate dimension_order (must be 5 chars, F-order)
    dimension_order = dimension_order.upper()
    valid_chars = set("XYZCT")
    if len(dimension_order) != 5 or set(dimension_order) != valid_chars:
        raise ValueError(
            f"dimension_order must be a permutation of 'XYZCT' (5 chars), "
            f"got '{dimension_order}'"
        )

    # Infer or validate np_index_order (C-order: first letter = axis 0)
    if np_index_order is None:
        # Backward compatible inference (C-order: first axis = first letter)
        if np_img.ndim == 2:
            np_index_order = "YX"
        elif np_img.ndim == 3:
            np_index_order = "YXC"
        elif np_img.ndim == 4:
            np_index_order = "ZYXC"
        elif np_img.ndim == 5:
            np_index_order = "TZYXC"
        else:
            raise ValueError(f"Cannot interpret data of dim {np_img.ndim}.")
    else:
        np_index_order = np_index_order.upper()
        if len(np_index_order) < 2 or len(np_index_order) > 5:
            raise ValueError(
                f"np_index_order must be 2-5 chars, got '{np_index_order}'"
            )
        if not set(np_index_order).issubset(valid_chars):
            raise ValueError(
                f"np_index_order must contain only chars from 'XYZCT', "
                f"got '{np_index_order}'"
            )
        if len(np_index_order) != len(set(np_index_order)):
            raise ValueError(
                f"np_index_order must not have duplicate chars, "
                f"got '{np_index_order}'"
            )
        if np_img.ndim != len(np_index_order):
            raise ValueError(
                f"np_index_order length ({len(np_index_order)}) "
                f"must match np_img.ndim ({np_img.ndim})"
            )

    # Build size dict from input array (np_index_order is C-order: axis i -> letter i)
    sizes = dict(zip(dimension_order, [1]*5))  # default size 1 for all dimensions
    for i, axis in enumerate(np_index_order):
        sizes[axis] = np_img.shape[i]

    byteorder = np_img.dtype.byteorder
    if byteorder == "=":
        import sys
        byteorder = "<" if sys.byteorder == 'little' else ">"

    endianness = 1 if byteorder == "<" else 0

    # Build the transpose: for each axis in desired output order, find its position in input
    # desired_order_c lists axes from axis 0 to axis -1 (C-order index order)
    desired_order_c = dimension_order[::-1]  # Reverse F-order to get C-order index order
    transpose_axes = [np_index_order.index(axis) for axis in desired_order_c if axis in np_index_order]
    np_img = np.transpose(np_img, axes=transpose_axes)

    # Ensure C-contiguous memory layout before tobytes()
    # transpose() creates a view with changed strides, but tobytes() on non-contiguous
    # arrays may produce bytes in unexpected order. ascontiguousarray() copies data
    # into C-contiguous layout (last axis varies fastest).
    np_img = np.ascontiguousarray(np_img)

    return Pixels(
        bindata = BinData(data=np_img.tobytes(), endianness=endianness),
        size_x = sizes['X'],
        size_y = sizes['Y'],
        size_c = sizes['C'],
        size_z = sizes['Z'],
        size_t = sizes['T'],
        dimension_order = dimension_order,
        dtype = _canonicalize_dtype(np_img.dtype.str),
        **kwargs,
    )


def deserialize_to_numpy(
    pixels: Pixels,
    *,
    singleton_t: bool = True,
    np_index_order: str = "ZYXC"
) -> np.ndarray:
    '''Convert protobuf Pixels to a numpy array.

    Args:
        pixels: protobuf data
    Keyword Args:
        singleton_t: DEPRECATED. Use np_index_order to control output dimensions.
        np_index_order: Numpy index order string describing which axis corresponds to which dimension.
            First letter corresponds to numpy axis 0, second to axis 1, etc.
            Must be 2-5 characters (a permutation of subset of "ZYXCT").
            Dimensions not in np_index_order are squeezed (must be singleton).
            Defaults to "ZYXC" (4D output, T squeezed for backward compatibility).

    Returns:
        Numpy array (C-contiguous) with shape matching np_index_order.
        The dtype and byteorder matches the input.
    '''
    # Deprecation warning for singleton_t
    if singleton_t is not True:
        warnings.warn(
            "singleton_t parameter is deprecated. Use np_index_order to control "
            "output dimensions (include T to preserve, exclude to squeeze).",
            DeprecationWarning,
            stacklevel=2
        )

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

    def _get_dtype(pixels: Pixels) -> np.dtype:
        dt = np.dtype(pixels.dtype)

        if pixels.bindata.endianness == BinData.Endianness.BIG:
            dt = dt.newbyteorder(">")
        else:
            dt = dt.newbyteorder("<")

        return dt

    # Validate np_index_order (2-5 chars, C-order)
    np_index_order = np_index_order.upper()
    valid_chars = set("ZYXCT")
    if len(np_index_order) < 2 or len(np_index_order) > 5:
        raise ValueError(
            f"np_index_order must be 2-5 chars, got '{np_index_order}'"
        )
    if not set(np_index_order).issubset(valid_chars):
        raise ValueError(
            f"np_index_order must contain only chars from 'ZYXCT', "
            f"got '{np_index_order}'"
        )
    if len(np_index_order) != len(set(np_index_order)):
        raise ValueError(
            f"np_index_order must not have duplicate chars, "
            f"got '{np_index_order}'"
        )

    # Get dimension sizes
    dims = dict(
        Z = pixels.size_z or 1,
        Y = pixels.size_y or 1,
        X = pixels.size_x or 1,
        C = pixels.size_c or 1,
        T = pixels.size_t or 1,
    )

    # Validate: dimensions not in output must be singleton
    for axis, size in dims.items():
        if axis not in np_index_order and size > 1:
            raise ValueError(
                f"Dimension {axis} has size {size} but is not in np_index_order "
                f"'{np_index_order}'. Cannot squeeze non-singleton dimension."
            )

    np_img = np.frombuffer(
        pixels.bindata.data,
        dtype=_get_dtype(pixels),
    )

    # The dimension_order (from proto) is F-order: first letter varies fastest.
    # Convert to C-order (numpy convention): reverse the string.
    dim_order_c = pixels.dimension_order[::-1].upper()

    # Build shape in buffer order (C-order, matching the dimension_order reversed)
    shape_orig = [dims[k] for k in dim_order_c]
    np_img = np_img.reshape(shape_orig)

    # Now transpose to match desired np_index_order (C-order: first letter = axis 0)
    # dim_order_c describes current layout, np_index_order describes desired layout
    # We need to permute from dim_order_c to np_index_order

    # Build full target order including dimensions to squeeze
    full_target_order = list(np_index_order)
    for axis in "ZYXCT":
        if axis not in full_target_order:
            full_target_order.append(axis)

    # Build transpose indices: for each axis in full_target_order, find its position in dim_order_c
    dim_orig = [dim_order_c.index(k) for k in full_target_order]
    np_img = np_img.transpose(dim_orig)

    # Squeeze dimensions not in np_index_order (they should be singleton)
    squeeze_axes = [i for i, axis in enumerate(full_target_order) if axis not in np_index_order]
    if squeeze_axes:
        for idx in sorted(squeeze_axes, reverse=True):
            np_img = np_img.squeeze(axis=idx)

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
