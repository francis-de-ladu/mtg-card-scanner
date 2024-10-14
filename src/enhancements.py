import cv2
import numpy as np


def unsharp_mask(image, kernel_size=21, sigma=1.0, strength=1.5):
    """Apply Unsharp Masking to sharpen an image."""
    # Step 1: Blur the image
    blurred = cv2.GaussianBlur(image, (kernel_size, kernel_size), sigma)

    # Step 2: Subtract the blurred image from the original
    sharpened = cv2.addWeighted(image, 1.0 + strength, blurred, -strength, 0)

    return blurred, sharpened


def laplacian_filter(image):
    """Apply Laplacian filter to sharpen an image."""
    # Step 1: Apply the Laplacian filter
    laplacian = cv2.Laplacian(image, cv2.CV_64F)

    # Step 2: Convert to a suitable type (uint8) and add edges to the original image
    laplacian = cv2.convertScaleAbs(laplacian)
    sharpened = cv2.add(image, laplacian)

    return laplacian, sharpened


def high_pass_filter(image, kernel_size=21):
    """Apply High-Pass Filter to sharpen an image."""
    # Step 1: Apply Gaussian blur to create a low-pass filter
    low_pass = cv2.GaussianBlur(image, (kernel_size, kernel_size), 0)

    # Step 2: Subtract the low-pass image from the original
    high_pass = cv2.subtract(image, low_pass)

    # Step 3: Add the high-pass image back to the original
    sharpened = cv2.add(image, high_pass)

    return high_pass, sharpened


def combined_filters(image):
    _, sharpened_unsharp = unsharp_mask(image, kernel_size=21)
    laplacian, _ = laplacian_filter(image)
    high_pass, _ = high_pass_filter(image, kernel_size=21)
    combined = cv2.add(cv2.add(sharpened_unsharp, laplacian), high_pass)

    # _, sharpened_unsharp = unsharp_mask(combined, kernel_size=21)
    # laplacian, _ = laplacian_filter(combined)
    # high_pass, _ = high_pass_filter(combined, kernel_size=21)
    # combined = cv2.add(cv2.add(sharpened_unsharp, laplacian), high_pass)

    return combined


# Example usage:
# Load an image
# image = cv2.imread('path_to_your_image.jpg')

# Sharpen the image using the different methods
# sharpened_unsharp = unsharp_mask(image)
# sharpened_laplacian = laplacian_filter(image)
# sharpened_high_pass = high_pass_filter(image)

# Display the results
# cv2.imshow('Original', image)
# cv2.imshow('Unsharp Mask', sharpened_unsharp)
# cv2.imshow('Laplacian Filter', sharpened_laplacian)
# cv2.imshow('High Pass Filter', sharpened_high_pass)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
