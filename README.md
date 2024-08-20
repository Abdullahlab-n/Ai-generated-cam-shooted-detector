//sample snippets//
import cv2
import numpy as np 

def preprocess_image(image_path):
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return gray 

def extract_features(image):
    # Texture analysis
    laplacian = cv2.Laplacian(image, cv2.CV_64F)
    texture_features = np.mean(np.abs(laplacian)) 

    # Color analysis
    color_hist = cv2.calcHist([image], [0], None, [256], [0, 256])
    color_features = np.mean(color_hist) 

    # Edge detection
    edges = cv2.Canny(image, 100, 200)
    edge_features = np.sum(edges) / (image.shape[0] * image.shape[1]) 

    return texture_features, color_features, edge_features 

def compare_images(image1_features, image2_features):
    # Simple comparison based on feature differences
    diff = np.abs(image1_features - image2_features)
    return diff 

if __name__ == "__main__":
    image1_path = "cat_real.jpg"
    image2_path = "cat_ai.jpg" 

    img1 = preprocess_image(image1_path)
    img2 = preprocess_image(image2_path) 

    features1 = extract_features(img1)
    features2 = extract_features(img2) 

    diff = compare_images(features1, features2) 

    # Larger differences might indicate higher probability of AI generation
    print(diff)


//sample program 2//
#openCV.file
import cv2
import numpy as np
from skimage.feature import local_binary_pattern

def detect_all(img):
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Texture lbp = local_binary_pattern(gray, P=8, R=1, method="uniform") # Lighting equ = cv2.equalizeHist(gray) avg_intensity = np.mean(gray) # Color color_hist = cv2.calcHist([img], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256]) color_moments = cv2.meanStdDev(img.reshape((-1, 3))) # Focus laplacian = cv2.Laplacian(gray, cv2.CV_64F) focus_measure = cv2.mean(cv2.abs(laplacian))[0] # Perspective edges = cv2.Canny(gray, 100, 200) lines = cv2.HoughLinesP(edges, 1, np.pi/180, 100, minLineLength=100, maxLineGap=10) # Background fgbg = cv2.createBackgroundSubtractorKNN() fgmask = fgbg.apply(img) # Noise fft = np.fft.fft2(gray) fft_shift = np.fft.fftshift(fft) # Edge Detection and Sharpness edges = cv2.Canny(gray, 100, 200) sharpness = np.var(laplacian)

# Artifacts
# Implement artifact detection logic based on specific artifacts

# Consistency
# Implement consistency checks based on specific criteria

# Add metadata (replace with actual metadata extraction logic)
metadata = {'width': img.shape[1], 'height': img.shape[0]}

return lbp, equ, avg_intensity, color_hist, color_moments, focus_measure, lines, fgmask, fft_shift, edges, sharpness, metadata
