import cv2
import numpy as np
import os

# Output directory to save the final mask image
OUT_DIR = 'results/exercise-6'
if not os.path.exists(OUT_DIR):
    os.makedirs(OUT_DIR)
    
def preprocess_image(image_path):
    ## 1. Read the input image and convert it to grayscale.
    # Grayscale simplifies the processing by removing color information,
    # keeping only intensity (luminance) which is sufficient for text segmentation.
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    ## 2. Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
    # CLAHE enhances local contrast, especially in regions with varying illumination.
    # This helps to better distinguish text from background.
    # clipLimit controls contrast amplification, and tileGridSize sets the size of the contextual regions.
    clahe = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)

    ## 3. Adaptive Thresholding (instead of Otsu)
    # Adaptive Gaussian Thresholding is used to better handle varying lighting conditions across the image.
    # It calculates the threshold locally for each pixel neighborhood.
    # blockSize=11 defines the neighborhood size, and C=2 is a constant subtracted from the mean.
    binary = cv2.adaptiveThreshold(
        enhanced, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
        cv2.THRESH_BINARY, 
        11, 2
    )

    ## 4. Remove small components (noise) using connected components analysis
    # This filters out tiny connected regions (e.g. dust or small spots) that are not part of the actual characters.
    # Setting min_area to 0 means all regions are kept, but this can be modified for stricter noise filtering.
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    min_area = 0 
    mask = np.zeros_like(binary)
    for i in range(1, num_labels):  # skip label 0 which is the background
        if stats[i, cv2.CC_STAT_AREA] >= min_area:
            mask[labels == i] = 255

    ## 5. Apply dilation to thicken the character strokes
    # Dilation helps to fill gaps inside the text and connect broken parts.
    # A very small kernel (1x1) is used to preserve thin strokes while slightly reinforcing the structure.
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 1))
    dilated = cv2.dilate(mask, kernel, iterations=2)

    ## 6. Invert the binary image
    # The final output should have white text (foreground) on a black background.
    final = cv2.bitwise_not(dilated)

    return final


def save_result(mask, output_path):
    # Save the final processed binary mask to the specified file path.
    cv2.imwrite(output_path, mask)
    print(f"Mask saved at {output_path}")

def main():
    input_path = 'img/book-cover.jpeg'
    output_path = os.path.join(OUT_DIR, 'book-mask.png')

    # Create output directory if it doesn't exist
    os.makedirs(OUT_DIR, exist_ok=True)

    # Run preprocessing and generate the final mask
    mask = preprocess_image(input_path)

    # Save the result
    save_result(mask, output_path)

if __name__ == '__main__':
    main()
