from typing import Tuple
from math import floor, ceil
import numpy as np
from .fft import DFT2, IDFT2, fft_shift


def find_padding(w_shape) -> Tuple[int, int]:
    padding = (w_shape - 1) / 2
    
    floored_padding = floor(padding)
    ceiled_padding = ceil(padding)
    # If the kernel has even dimensions, padding will be unequal.
    # In this case, choosing the higher index as the center.
    if floored_padding < padding:
        # Consider the right or bottom pixel as the center
        padding_lower = ceiled_padding
        # 2*padding is guaranteeded to be an int
        padding_higher = int(2*padding) - padding_lower
    else:
        padding_lower = floored_padding
        padding_higher = floored_padding
    
    return padding_lower, padding_higher


def fft_pad(w_shape):
    padding_lower = w_shape
    padding_higher = w_shape

    return padding_lower, padding_higher


def pad_image(image: np.ndarray, padding_h_lower, 
              padding_h_higher, padding_w_lower, padding_w_higher, padding_type, flatten=True) -> np.ndarray:
    """
    image: Image to pad
    w: kernel
    padding_type: type of padding to apply
    flatten: Whether to remove the dummy dimension for greyscale image
    """
    
    # If greyscale, add a dmmy dimension for convenience
    if np.ndim(image) == 2:
        image_copy = np.expand_dims(image, axis=2)
    else:
        image_copy = image.copy()
    
    H, W, C = image_copy.shape
    
    total_padding_h = padding_h_lower + padding_h_higher
    
    total_padding_w = padding_w_lower + padding_w_higher
    
    # Initialize with a zero-padded image
    padded_image = np.zeros((H+total_padding_h, W+total_padding_w, C), dtype=image.dtype)
    padded_image[padding_h_lower:padding_h_lower+H, padding_w_lower:padding_w_lower+W, :] = image_copy
    
    if padding_type == "zero":
        # Already padded, do nothing
        pass
    elif padding_type == "copy-edge":
        left_edge = image_copy[:, [0], :]
        right_edge = image_copy[:, [-1], :]
        padded_image[padding_h_lower:padding_h_lower+H, :padding_w_lower, :] = left_edge
        padded_image[padding_h_lower:padding_h_lower+H, padding_w_lower+W:, :] = right_edge
        
        top_edge = padded_image[[padding_h_lower], :, :]
        bottom_edge = padded_image[[padding_h_lower+H-1], :, :]
        padded_image[:padding_h_lower, :, :] = top_edge
        padded_image[padding_h_lower+H:, :, :] = bottom_edge
    
    elif padding_type == 'reflect-edge':
        left_edge = padded_image[:, padding_w_lower:2*padding_w_lower, :]
        inverted_left_edge = left_edge[:, ::-1, :]
        right_edge = padded_image[:, -2*padding_w_higher:-padding_w_higher, :]
        inverted_right_edge = right_edge[:, ::-1, :]
        padded_image[:, :padding_w_lower, :] = inverted_left_edge
        padded_image[:, padding_w_lower+W:, :] = inverted_right_edge
        
        top_edge = padded_image[padding_h_lower:2*padding_h_lower, :, :]
        inverted_top_edge = top_edge[::-1, :, :]
        bottom_edge = padded_image[-2*padding_h_higher:-padding_h_higher, :, :]
        inverted_bottom_edge = bottom_edge[::-1, :, :]
        padded_image[:padding_h_lower, :, :] = inverted_top_edge
        padded_image[padding_h_lower+H:, :, :] = inverted_bottom_edge
        
    elif padding_type == "wrap-around":
        left_edge = padded_image[:, padding_w_lower:2*padding_w_lower, :]
        right_edge = padded_image[:, -2*padding_w_higher:-padding_w_higher, :]
        padded_image[:, padding_w_lower+W:, :] = left_edge
        
        top_edge = padded_image[padding_h_lower:2*padding_h_lower, :, :]
        bottom_edge = padded_image[-2*padding_h_higher:-padding_h_higher, :, :]
        padded_image[:padding_h_lower, :, :] = bottom_edge
        padded_image[padding_h_lower+H:, :, :] = top_edge
    else:
        raise ValueError(f"Unsupported padding type {padding_type}.")
        
    if padded_image.shape[2] == 1 and flatten:
        padded_image = padded_image[:, :, 0]
        
    return padded_image


def im2col(padded_image, kernel_size, original_size):
    """Convert image to a matrix representation

    Args:
        image (np.ndarray): Padded image
        kernel_size (Tuple[int, int]): Kernel size
        original_size (Tuple[int, int]): Dimensions of the original image
    """    
    kernel_h, kernel_w = kernel_size

    H, W = original_size

    row_indices = np.tile(np.repeat(np.arange(kernel_h), kernel_w), W).reshape(W, kernel_h*kernel_w).T
    col_indices = np.array([np.tile(np.arange(kernel_w), kernel_h) + offset for offset in range(W)]).T
    
    row_indices = np.concatenate([row_indices + offset for offset in range(H)], axis=1)
    col_indices = np.tile(col_indices, H)
    
    return padded_image[row_indices, col_indices]


def col2im(image, H, W):
    return image.reshape(H, W)


def conv2_matmul(f: np.ndarray, w: np.ndarray, pad: str) -> np.ndarray:
    
    """
    Convolution by reshaping the image to column format. Assumes grayscale image.
    """

    f_copy = f.copy().astype(np.float32)

    H, W = f_copy.shape

    (padding_h_lower, padding_h_higher), (padding_w_lower, padding_w_higher) \
                                        = (find_padding(w_shape) for w_shape in w.shape)

    padded_image = pad_image(f_copy, padding_h_lower, padding_h_higher, padding_w_lower, padding_w_higher, pad, flatten=True)

    reshaped_kernel = w.reshape(1, -1)

    column_image = im2col(padded_image, w.shape, (H, W))
    conv_result = np.dot(reshaped_kernel, column_image)

    convolved_image = conv_result.reshape(H, W)

    return convolved_image


def pad_kernel(kernel, expected_H, expected_W):
    # Zero pad the kernel to the same size as that of the image
    required_padding_h = expected_H - kernel.shape[0]
    required_padding_w = expected_W - kernel.shape[1]

    padding_left = required_padding_w // 2
    padding_right = required_padding_w - padding_left

    padding_top = required_padding_h // 2
    padding_bottom = required_padding_h - padding_top

    # Pad the kernel to be the same size as the input
    padded_kernel = pad_image(kernel, padding_top, padding_bottom, padding_left, padding_right, "zero", flatten=True)
    return padded_kernel


def fft_conv(f, w, pad):

    H, W = f.shape[:2]

    # Pad the kernel to the nearest greater odd size
    kernel_H, kernel_W = w.shape[:2]
    kernel = w.copy()
    if kernel_H % 2 == 0:
        padding = np.zeros((1, kernel_W))
        kernel = np.concatenate([padding, kernel], axis=0)
        kernel_H += 1
    if kernel_W % 2 == 0:
        padding = np.zeros((kernel_H, 1))
        kernel = np.concatenate([padding, kernel], axis=1)
        kernel_W += 1

    # Find the padding required for linear convolution using DFT
    (padding_h_lower, padding_h_higher), (padding_w_lower, padding_w_higher) \
                                        = (fft_pad(w_shape) for w_shape in kernel.shape)

    padded_image = pad_image(f, padding_h_lower, padding_h_higher, padding_w_lower, padding_w_higher, pad, flatten=True)

    # Zero pad the kernel to the same size as that of the image
    padded_kernel = pad_kernel(kernel, padded_image.shape[0], padded_image.shape[1])

    # Calculate the DFT
    dft_image = DFT2(padded_image)
    dft_kernel = DFT2(padded_kernel)

    # conv in time -> pointwise multiplication in frequency
    pointwise_mul = dft_image * dft_kernel

    # Inverse FT will give the convolution result
    idft_result = IDFT2(pointwise_mul)

    conv_result = fft_shift(idft_result).real

    left, top = kernel_W - 1, kernel_H - 1

    return conv_result[top:top+H, left:left+W]


def conv2(f: np.ndarray, w: np.ndarray, pad: str, trunc: bool=True, use_dft=False) -> np.ndarray:
    
    """
    Function which performs the convolution. The padding is handled internally
    based on the kernel dimensions.
    """

    f_copy = f.copy().astype(np.float32)

    # If grayscale
    if np.ndim(f_copy) == 2:
        if max(w.shape) > 7 or use_dft:
            # Convolution using DFT
            convolved_image = fft_conv(f_copy, w, pad)
        else:
            # Standard linear convolution
            convolved_image = conv2_matmul(f_copy, w, pad)
    # If RGB
    else:
        
        convolved_image_channels = []
        C = f_copy.shape[-1]
        for c in range(C):
            if max(w.shape) > 7 or use_dft:
                # Convolution using DFT
                convolved_image = fft_conv(f_copy[:, :, 0], w, pad)
            else:
                # Standard linear convolution
                convolved_image = conv2_matmul(f_copy, w, pad,)
            convolved_image_channels.append(np.expand_dims(convolved_image, axis=2))
        convolved_image = np.concatenate(convolved_image_channels, axis=2)

    if trunc:
        convolved_image = convolved_image.astype(np.uint8)
    return convolved_image
