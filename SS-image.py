import numpy as np
import cv2

def load_image(file_path, grayscale=True):
    """
    Load image from file.

    Args:
        file_path (str): Path to the image file.
        grayscale (bool): Whether to load the image as grayscale. Default is True.

    Returns:
        np.ndarray: Loaded image.
    """
    if grayscale:
        return cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
    else:
        return cv2.imread(file_path)

def smooth(image, kernel_size=(6, 6)):
    """
    Apply smoothing (average filter) to the image.

    Args:
        image (np.ndarray): Input image.
        kernel_size (tuple): Size of the kernel for smoothing. Default is (3, 3).

    Returns:
        np.ndarray: Smoothed image.
    """
    kernel = np.ones(kernel_size, np.float32) / (kernel_size[0] * kernel_size[1])
    smoothed = cv2.filter2D(image, -1, kernel)
    return smoothed

def sharpen(image, alpha=2, beta=-0.7):
    """
    Apply sharpening to the image.

    Args:
        image (np.ndarray): Input image.
        alpha (float): Weight of the original image. Default is 2.5.
        beta (float): Weight of the blurred image. Default is -1.

    Returns:
        np.ndarray: Sharpened image.
    """
    blurred = cv2.GaussianBlur(image, (3, 3), 0)
    sharpened = cv2.addWeighted(image, alpha, blurred, beta, 0)
    return sharpened

def display_images(image_dict):
    """
    Display images using OpenCV.

    Args:
        image_dict (dict): Dictionary containing image names as keys and corresponding images as values.
    """
    for name, img in image_dict.items():
        cv2.imshow(name, img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# Load the image
image = load_image('image.jpg')

# Apply smoothing
smoothed_image = smooth(image)

# Apply sharpening
sharpened_image = sharpen(image)

# Display the results
display_images({
    'Original Image': image,
    'Smoothed Image': smoothed_image,
    'Sharpened Image': sharpened_image
})
