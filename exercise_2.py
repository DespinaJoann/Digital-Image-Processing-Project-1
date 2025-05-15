import os
import numpy as np
import matplotlib.pyplot as plt
import cv2

OUT_DIR = 'results/exercise-2'

# Create the output directory if it doesn't exist
if not os.path.exists(OUT_DIR):
    os.makedirs(OUT_DIR)
    
def save_spectrum(spectrum, filename, cmap='gray'):
    """Sabe the spectrum as an image."""
    plt.imshow(spectrum, cmap=cmap) 
    plt.axis('off')
    plt.savefig(filename, bbox_inches='tight', pad_inches=0)
    plt.close()

def main():
    ## 1. Load the image and convert to grayscale
    image = cv2.imread('img/cornfield.jpg', cv2.IMREAD_GRAYSCALE)
    
    
    ## 2. Calculate the 2D DFT of the image
    f = np.fft.fft2(image)                     # Compute the 2D Fourier Transform
    fshift = np.fft.fftshift(f)                # Shift zero frequency to the center of the spectrum
   
    
    ## 3. Calculate the magnitude and phase spectrums
    magnitude = np.abs(fshift)
    phase = np.angle(fshift)
    
    # Log scale magnitude for visualization (for better visibility)
    log_magnitude = np.log(magnitude + 1)     
    
    # Save original magnitude and phase spectra
    save_spectrum(log_magnitude, os.path.join(OUT_DIR, 'magnitude_spectrum.png'))
    save_spectrum(phase, os.path.join(OUT_DIR, 'phase_spectrum.png'), cmap='twilight') # twilight is a cyclic colormap, appropriate for phase visualization


    ## 4. Modify the phase spectrum: vertical flip (symmetry over horizontal axis)
    flipped_phase = np.flip(phase, axis=0)  
    
    # Combine original magnitude with flipped phase
    modified_fshift = magnitude * np.exp(1j * flipped_phase)
    
    # Visualization of modified spectra
    mod_log_magnitude = np.log(np.abs(modified_fshift) + 1) 
    
    # Save modified magnitude and phase spectra
    save_spectrum(mod_log_magnitude, os.path.join(OUT_DIR, 'modified_magnitude_spectrum.png'))
    save_spectrum(flipped_phase, os.path.join(OUT_DIR, 'modified_phase_spectrum.png'), cmap='twilight')
   
   
    ## 5. Inverse DFT to reconstruct the image
    inv_shift = np.fft.ifftshift(modified_fshift)      # Shift back
    reconstructed_image = np.fft.ifft2(inv_shift)      # Inverse FFT
    reconstructed_image = np.abs(reconstructed_image)  # Get the real part

    # Normalize and save the reconstructed image
    reconstructed_image = cv2.normalize(reconstructed_image, None, 0, 255, cv2.NORM_MINMAX)
    reconstructed_image = np.uint8(reconstructed_image)
    cv2.imwrite(os.path.join(OUT_DIR, 'reconstructed_image.jpg'), reconstructed_image)

    ## 6. Get the Combined View
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.imshow(log_magnitude, cmap='gray')
    plt.title('Original Magnitude Spectrum')
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.imshow(phase, cmap='twilight')
    plt.title('Original Phase Spectrum')
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, 'original_spectra_combined.png'))
    plt.close()
    
if __name__ == '__main__':
    main()
