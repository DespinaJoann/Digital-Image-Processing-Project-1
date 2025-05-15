
import os
import numpy as np
import cv2 
import matplotlib.pyplot as plt

OUT_DIR = 'results/exercise-3'
# Create the output directory if it doesn't exist
if not os.path.exists(OUT_DIR):
    os.makedirs(OUT_DIR)

def sobel_edge_detection(image_path):
    """Apply Sobel edge detection to the image."""
    # Read the image in grayscale
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    
    # Apply Sobel filter for edge detection (horizontal and vertical)
    sobel_x = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=5)  # Sobel filter in x direction
    sobel_y = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=5)  # Sobel filter in y direction
    sobel_edges = cv2.magnitude(sobel_x, sobel_y)          # Combine the two gradients
    
    # Normalize for visualization
    sobel_edges = cv2.normalize(sobel_edges, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    
    # Save image
    plt.imsave(os.path.join(OUT_DIR, 'sobel_girlface.png'), sobel_edges, cmap='gray')
    
    
def canny_edge_detection(image_path, low_threshold, high_threshold):
    """Apply Canny edge detection to the image."""
    # Read the image in grayscale
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    
    # Apply Canny filter to detect edges and outlines
    canny_edges = cv2.Canny(image, low_threshold, high_threshold) # low threshold is for edge linking, high threshold is for edge detection
    
    # Save the Canny edge-detected image (grayscale)
    plt.imsave(os.path.join(OUT_DIR, 'canny_fruits.png'), canny_edges, cmap='gray')
        
def loG_filter(image_path, sigma):
    """Apply Laplacian of Gaussian (LoG) filter to the image."""
    # Read the image in grayscale
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    
    # Apply Gaussian blur filter for smoothing/bluring of the image
    # This is a pre-processing step before applying the Laplacian filter
    # The sigma parameter controls the amount of blurring
    # A larger sigma results in more blurring
    # The kernel size must be odd and positive
    # The kernel size is determined by the sigma value
    # A common choice is to use a kernel size of 5x5 for sigma = 1.0
    blurred_image = cv2.GaussianBlur(image, (5, 5), sigma)
    
    # Apply Laplacian filter for detecting details in the image
    log_edges = cv2.Laplacian(blurred_image, cv2.CV_64F)
    
    # Normalize for visualization
    log_edges = cv2.normalize(log_edges, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    
    # Save the LoG edge-detected image (grayscale)
    plt.imsave(os.path.join(OUT_DIR, 'log_leaf.png'), log_edges, cmap='gray')
        
        
def main():
    ## Path to the input images
    img_path_1 = 'img/girlface.jpg'
    img_path_2 = 'img/fruits.jpg'
    img_path_3 = 'img/leaf.jpg'
    
    ## 1. Apply Sobel edge detection
    sobel_edge_detection(img_path_1)
    
    ## 2. Apply Canny edge detection with specified thresholds
    # low_threshold has to be less than high_threshold
    # The low threshold is used for edge linking, and the high threshold is used for edge detection
    # The Canny edge detector uses two thresholds to detect strong and weak edges
    # The strong edges are the ones above the high threshold
    # The weak edges are the ones below the low threshold
    # The weak edges are only considered if they are connected to strong edges
    # So having a low threshold is important to detect weak edges
    # The high threshold is used to detect strong edges
    # And 100 and 200 are common values for low and high thresholds respectively
    canny_edge_detection(img_path_2, low_threshold=100, high_threshold=200)
    
    ## 3. Apply Laplacian of Gaussian (LoG) filter with specified sigma
    loG_filter(img_path_3, sigma=1.0)
    
if __name__ == "__main__":
    main()