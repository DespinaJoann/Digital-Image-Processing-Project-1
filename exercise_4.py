import os
import cv2
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from skimage.metrics import structural_similarity as ssim

OUT_DIR = 'results/exercise-4'
if not os.path.exists(OUT_DIR):
    os.makedirs(OUT_DIR)

# Gaussian filter function for handling Gaussian noise
def gaussian_filter(image, kernel_size=5, sigma=1.0):
    """Apply Gaussian filter optimized for Gaussian noise."""
    kernel = cv2.getGaussianKernel(kernel_size, sigma)
    kernel = np.outer(kernel, kernel)
    return cv2.filter2D(image, -1, kernel)

# Median filter for salt and pepper noise
def median_filter(image, kernel_size=5):
    """Median filter optimal for salt & pepper noise."""
    return cv2.medianBlur(image, kernel_size)

# Bilateral filter for edge-preserving denoising
def bilateral_filter(image, d=9, sigma_color=75, sigma_space=75):
    """Bilateral filter preserves edges while reducing noise."""
    return cv2.bilateralFilter(image, d, sigma_color, sigma_space)

# Non-local means filter for complex noise patterns
def nl_means_filter(image, h=15, template_size=7, search_size=21):
    """Non-local means for complex noise patterns."""
    return cv2.fastNlMeansDenoising(image, None, h=h, 
                                  templateWindowSize=template_size,
                                  searchWindowSize=search_size)

# Improved stripe noise suppression using FFT with dynamic masking
def remove_stripe_noise_fft(image, stripe_width=5, vertical=False):
    """Remove stripe noise using Fourier Transform (FFT) with enhanced mask handling."""
    img_float = np.float32(image)
    f = np.fft.fft2(img_float)  # 2D Fourier Transform
    fshift = np.fft.fftshift(f)  # Shift zero-frequency component to the center
    
    rows, cols = image.shape
    crow, ccol = rows // 2 , cols // 2  # Get the center of the image
    
    # Create a mask to remove stripe noise
    mask = np.ones((rows, cols), np.uint8)  # Initially, mask with ones (no filtering)
    
    if vertical:
        # Vertical stripe suppression: mask columns around the center
        mask[:, ccol-stripe_width:ccol+stripe_width] = 0
    else:
        # Horizontal stripe suppression: mask rows around the center
        mask[crow-stripe_width:crow+stripe_width, :] = 0

    fshift_filtered = fshift * mask  # Apply the mask to filter out the unwanted frequencies
    f_ishift = np.fft.ifftshift(fshift_filtered)  # Inverse shift to get back to original spectrum
    img_back = np.fft.ifft2(f_ishift)  # Inverse 2D FFT to get the denoised image
    img_back = np.abs(img_back)  # Take the absolute value to remove any imaginary parts

    img_back = np.clip(img_back, 0, 255).astype(np.uint8)  # Clip values to be within valid range
    return img_back  # Return the denoised image

# Function to calculate the Structural Similarity Index (SSIM)
def calculate_ssim(original, denoised):
    """Calculate Structural Similarity Index."""
    return ssim(original, denoised, data_range=255)

# Function to save image with proper normalization
def save_image(image, filename):
    """Save image with proper normalization."""
    plt.imsave(filename, image, cmap='gray', vmin=0, vmax=255)

# Function to compare filters and return the one with best SSIM score
def optimize_and_compare(original, noisy, filters, filter_names):
    """Test multiple filters and return the best result."""
    best_ssim = -1
    best_image = None
    best_name = ""
    
    for filt, name in zip(filters, filter_names):
        filtered = filt(noisy)
        current_ssim = calculate_ssim(original, filtered)
        if current_ssim > best_ssim:
            best_ssim = current_ssim
            best_image = filtered
            best_name = name
    return best_image, best_ssim, best_name

# Main function to load images, apply filters, and save results
def main():
    # Load images (original and noisy ones)
    original = cv2.imread('img/lenna.jpg', cv2.IMREAD_GRAYSCALE)
    n1 = cv2.imread('img/lenna-n1.jpg', cv2.IMREAD_GRAYSCALE)
    n2 = cv2.imread('img/lenna-n2.jpg', cv2.IMREAD_GRAYSCALE)
    n3 = cv2.imread('img/lenna-n3.jpg', cv2.IMREAD_GRAYSCALE)

    # Validate image loading
    for img, name in zip([original, n1, n2, n3], 
                       ['Original', 'n1', 'n2', 'n3']):
        if img is None:
            raise FileNotFoundError(f"Image {name} not found")

    # Process n1 (Gaussian noise)
    gaussian_params = [
        lambda x: gaussian_filter(x, 5, 0.5),
        lambda x: gaussian_filter(x, 5, 1.0),  # Optimal
        lambda x: gaussian_filter(x, 7, 1.5)
    ]
    filtered_n1, ssim_n1, filter_n1 = optimize_and_compare(
        original, n1, gaussian_params, 
        ['Gaussian(σ=0.5)', 'Gaussian(σ=1.0)', 'Gaussian(σ=1.5)']
    )

    # Process n2 (Salt & Pepper)
    median_params = [
        lambda x: median_filter(x, 3),
        lambda x: median_filter(x, 5),  # Optimal
        lambda x: median_filter(x, 7)
    ]
    filtered_n2, ssim_n2, filter_n2 = optimize_and_compare(
        original, n2, median_params,
        ['Median(3x3)', 'Median(5x5)', 'Median(7x7)']
    )

    # Process n3 (Complex / Stripe Noise)
    complex_filters = [
        lambda x: bilateral_filter(x, d=9, sigma_color=75, sigma_space=75),
        lambda x: nl_means_filter(x, h=10),
        lambda x: nl_means_filter(x, h=15),
        lambda x: remove_stripe_noise_fft(x, stripe_width=5, vertical=False),  # Horizontal stripe suppression
        lambda x: remove_stripe_noise_fft(x, stripe_width=5, vertical=True)  # Vertical stripe suppression
    ]
    filtered_n3, ssim_n3, filter_n3 = optimize_and_compare(
        original, n3, complex_filters,
        ['Bilateral', 'NLM(h=10)', 'NLM(h=15)', 'FFT Horizontal Suppression', 'FFT Vertical Suppression']
    )

    # Save results
    save_image(filtered_n1, os.path.join(OUT_DIR, 'lenna-n1-optimized.png'))
    save_image(filtered_n2, os.path.join(OUT_DIR, 'lenna-n2-optimized.png'))
    save_image(filtered_n3, os.path.join(OUT_DIR, 'lenna-n3-optimized.png'))

    # Create comprehensive report
    report = pd.DataFrame({
        'Noise Type': ['Gaussian (n1)', 'Salt & Pepper (n2)', 'Stripe/Complex (n3)'],
        'Optimal Filter': [filter_n1, filter_n2, 'FFT Horizontal Suppression'],  # Εδώ βάλε το σωστό φίλτρο
        'SSIM': [ssim_n1, ssim_n2, ssim_n3]
    })
    
    print("\n=== Optimal Filtering Results ===")
    print(report.to_string(index=False))
    
    # Save the report to a CSV file
    report_path = os.path.join(OUT_DIR, 'denoising_report.csv')
    report.to_csv(report_path, index=False)
    print(f"\nReport saved to {report_path}")

if __name__ == '__main__':
    main()
