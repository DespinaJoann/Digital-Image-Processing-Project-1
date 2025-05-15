
import os 
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.cluster import KMeans
from skimage.io import imread
from skimage.metrics import mean_squared_error as mse

OUT_DIR = 'results/exercise-1'
# Create the output directory if it doesn't exist
if not os.path.exists(OUT_DIR):
    os.makedirs(OUT_DIR)

def quantize_image(image, n_colors, pixels=None):
    # Apply KMeans clustering
    kmeans = KMeans(n_clusters=n_colors, random_state=42)
    kmeans.fit(pixels)
    
    # Replace each pixel with its corresponding cluster center
    quantized_pixels = kmeans.cluster_centers_[kmeans.labels_]
    
    # Reshape back to the original image shape and return    
    return quantized_pixels.reshape(image.shape).astype(np.uint8)


def main():
    ## 1. Load the image
    image = imread('img/flowers.jpg')
    original_shape = image.shape
    # Reshape the image to a 2D array of pixels 
    # Going from (height, width, channels) to (num_pixels = height * width, channels)
    pixels = image.reshape(-1, 3)

    ## 2. Define the levels of quantization 
    n_colors_list = [5, 20, 200, 1000]
    mse_values = []
    
    ## 3. Process the image for each level of quantization
        
    for k in n_colors_list:
        # Quantize the image
        quantized_image = quantize_image(image, k, pixels)
        
        # Calculate MSE
        mse_value = mse(image, quantized_image)
        mse_values.append(mse_value)
        
        # Save the quantized image
        quantized_image_path = os.path.join(OUT_DIR, f'quantized_{k}_colors.jpg')
        plt.imsave(quantized_image_path, quantized_image)
        

    ## 4. Construct the dataframe containing MSE values vs Colors
    df = pd.DataFrame({
        'Colors': n_colors_list,
        'MSE': mse_values
    })

    # Save the dataframe to a CSV file
    csv_path = os.path.join(OUT_DIR, 'mse_values.csv')
    df.to_csv(csv_path, index=False)

    # Plot the MSE values
    plt.figure(figsize=(10, 6))
    plt.plot(df['Colors'], df['MSE'], marker='o')
    plt.title('MSE vs Number of Colors')
    plt.xlabel('Number of Colors')
    plt.ylabel('Mean Squared Error (MSE)')
    plt.xscale('log')
    plt.grid(True)
    plt.savefig(os.path.join(OUT_DIR, 'mse_vs_colors.png'))
    

    
if __name__ == "__main__":
    main()