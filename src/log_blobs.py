import time
import numpy as np
import cv2
import matplotlib.pyplot as plt
from .convolution import conv2, im2col, pad_image, col2im


# Some generic constants
scale_factor = np.sqrt(2.) # k - the scaling factor for the scale space

def lofg2(sigma):
    truncate = 3.0
    sd = float(sigma)
    width = float(int(truncate * sd + 0.5))
    size = int(2*width + 1)

    x = np.linspace(-width, width, size)
    y = np.linspace(-width, width, size)
    
    xx, yy = np.meshgrid(x, y)
    
    sigma2 = sigma**2
    gaussian = np.exp(-(xx**2 + yy**2)/(2*sigma2))
    kernel = (((xx**2 + yy**2) - 2*sigma2)/(sigma2*2)) * gaussian
    kernel = kernel / (np.pi*sigma2**2)
    
    return kernel


def gaussian_filter(image: np.ndarray, sigma: float) -> np.ndarray:
    """Smoothens the image using Gausian kernel with std=sigma

    Args:
        image (np.ndarray): Image to smoothen
        sigma (float): Standard Deviation of the kernel
    """
    # Determine the width of the kernel
    radius = int(4.*sigma + 0.5)
    x = np.linspace(-radius, radius, 2*radius)

    # generate the kernel and normalize
    kernel = (1. / np.sqrt(2*np.pi*sigma**2))*(np.exp(-x**2/(2*sigma**2)))
    kernel = kernel / np.sum(kernel)

    # Separable conv
    _filtered_x = conv2(image, kernel[None, :], pad="zero", trunc=False)
    filtered_image = conv2(_filtered_x, kernel[:, None], pad="zero", trunc=False)

    return filtered_image

def generate_dog_scalespace(image, sigma, n, k=scale_factor):
    """Generates the Laplacian scale space approximated with DoG

    Args:
        image (np.ndarray): Image
        sigma (float): Lowest scale
        n (int): height if the scale space

    Returns:
        List[np.ndarray]: Laplacian Scale space
    """    

    scale = sigma
    smoothed_image_upper = gaussian_filter(image, scale)
    _image = smoothed_image_upper.copy()

    scale_space = []

    for i in range(n):
        scale = k * scale
        smoothed_image_lower = gaussian_filter(_image, scale)
        dog =  (smoothed_image_lower - smoothed_image_upper)**2
        scale_space.append(dog)

        smoothed_image_upper = smoothed_image_lower.copy()

    return np.stack(scale_space)


def generate_log_scalespace(image, sigma, n, k=scale_factor):
    """Generates the Laplacian scale space approximated with DoG

    Args:
        image (np.ndarray): Image
        sigma (float): Lowest scale
        n (int): height if the scale space

    Returns:
        List[np.ndarray]: Laplacian Scale space
    """    

    _image = image.astype(np.float32)/255.
    _image = - _image

    scale = sigma
    scale_space = []
    for i in range(n):
        kernel = lofg2(scale) * (scale**2)
        # Normalized response
        log_response = conv2(_image, kernel, pad="zero", trunc=False)
        sq_log =  log_response**2
        scale_space.append(sq_log)
        scale = scale * k

    return np.stack(scale_space)


def nms2D(image, threshold):

    H, W = image.shape[:2]
    # Use Copy Edge padding to ignore the spurious maxima at the boundaries
    padded_image = pad_image(image, 1, 1, 1, 1, "copy-edge", flatten=True)
    column_image = im2col(padded_image, (3, 3), (H, W))

    # 4 is the index where the center of the 3x3 "kernel" will be at when flattened
    centers = column_image[4, :]
    max_values = np.argmax(column_image, axis=0)

    is_maxima = np.logical_and(max_values == 4, centers > threshold)

    maxima_image = col2im(is_maxima, H, W)

    return maxima_image


def nms(scale_space, threshold):

    D, H, W = scale_space.shape

    maxima_images = []
    for d in range(1, D-1):
        maxima_image = nms2D(scale_space[d, :, :], threshold=threshold)
        maxima_images.append(maxima_image)
    maxima_images = np.stack(maxima_images)
    padding_response = np.zeros_like(maxima_image)[None, :, :]
    maxima_images = np.concatenate([padding_response, maxima_images, padding_response], axis=0)

    maxima_points_2d = np.nonzero(maxima_images)

    zz, yy, xx = maxima_points_2d

    nms_points = np.zeros_like(maxima_images)
    # Only compare the xy locations which have potential maxima
    for z, y, x in zip(zz, yy, xx):

        lower_d, upper_d = z-1, z+1
        lower_i, upper_i = y-1, y+1
        lower_j, upper_j = x-1, x+1

        # Max value in the lower and upper scales surrounding the center
        max_value = max(
            np.max(scale_space[lower_d, lower_i:upper_i+1, lower_j:upper_j+1]), 
            np.max(scale_space[upper_d, lower_i:upper_i+1, lower_j:upper_j+1])
        )
        # Maxima only if the max value is greater than all the values in the
        # neighborhood
        is_maxima = scale_space[z, y, x] > max_value

        nms_points[z, y, x] = is_maxima

    return nms_points


def _draw_circles(image, row_idx, col_idx, radius):

    for x, y in zip(col_idx, row_idx):
        image = cv2.circle(image, (x, y), radius, (255, 0, 0))

    return image


def draw_circles(image, nms_result, sigma, k, show=False, save=False, filename=None):

    if np.ndim(image) == 2 or image.shape[-1] == 1:
        _image = image[:, :].copy()
        _image = _image[:, :, None]
        image_draw = np.concatenate([_image, _image, _image], axis=2)
    else:
        image_draw = image.copy()
    
    scale = sigma
    for scale_map in nms_result:
        row_idx, col_idx = np.nonzero(scale_map)
        radius = int(scale * np.sqrt(2))
        image_draw = _draw_circles(image_draw, row_idx, col_idx, radius)
        scale = scale * k

    if save:
        assert filename is not None
        cv2.imwrite(filename, image_draw[:, :, ::-1])

    if show:
        fig = plt.figure(dpi=100)
        plt.imshow(image_draw)
        plt.xticks([])
        plt.yticks([])
        plt.show()
    return image_draw


def detect_bolbs(image, sigma, n, k=scale_factor, threshold=0.01, use_dog=False):
    
    start_time = time.time()
    if use_dog:
        scale_space = generate_dog_scalespace(image=image, sigma=sigma, n=n+1, k=k)
        print(f"Scale space generated in {time.time() - start_time} seconds.")
    else:
        # Generate 2 extra scales
        _sigma = sigma / k
        scale_space = generate_log_scalespace(image=image, sigma=_sigma, n=n+2, k=k)
        print(f"Scale space generated in {time.time() - start_time} seconds.")

    start_time_nms = time.time()

    # Do NMS and remove the extra scales
    blob_centers = nms(scale_space, threshold=threshold)[1:-1]
    end_time = time.time()
    print(f"NMS done in {end_time - start_time_nms} seconds.")
    print(f"Total time required: {end_time - start_time} seconds.")

    return blob_centers
    