import os
import cv2
import numpy as np


OUT_DIR = 'results/exercise-5'
os.makedirs(OUT_DIR, exist_ok=True)

def main():
    ## 1. Load the image
    image = cv2.imread('img/parking-lot.jpg')
    img_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    ## 2. Threshold (binary inverse)
    _, thresh = cv2.threshold(img_gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    ## 3. Remove noise
    kernel = np.ones((3,3), np.uint8)
    opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)

    ## 4. Sure background area
    sure_bg = cv2.dilate(opening, kernel, iterations=3)

    ## 5. Finding sure foreground area (distance transform)
    dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 5)
    ret, sure_fg = cv2.threshold(dist_transform, 0.4 * dist_transform.max(), 255, 0)

    ## 6. Unknown region
    sure_fg = np.uint8(sure_fg)
    unknown = cv2.subtract(sure_bg, sure_fg)

    ## 7. Marker labeling
    _, markers = cv2.connectedComponents(sure_fg)

    ## 8. Add 1 to all labels so that sure background is not 0, but 1
    markers = markers + 1

    ## 9. Mark the region of unknown with zero
    markers[unknown == 255] = 0

    ## 10.  Watershed
    markers = cv2.watershed(image, markers)

    img_result = image.copy()
    img_result[markers == -1] = [0,0,255]  # Borders in red

    # 11. Detect bounding boxes
    bounding_boxes = []
    for label in np.unique(markers):
        if label <= 1:
            continue
        mask = np.uint8(markers == label)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            c = max(contours, key=cv2.contourArea)
            x, y, w, h = cv2.boundingRect(c)
            if w > 10 and h > 10:
                bounding_boxes.append((x, y, w, h))
                cv2.rectangle(img_result, (x, y), (x+w, y+h), (0, 255, 0), 2)

    print(f"Detected cars: {len(bounding_boxes)}")

    ## 12. Save the image with bounding boxes
    cv2.imwrite(os.path.join(OUT_DIR, 'parking-lot-detected-cars-result.jpg'), img_result)


if __name__ == "__main__":
    main()