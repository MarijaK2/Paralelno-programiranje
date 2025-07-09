import cv2
import numpy as np
import matplotlib.pyplot as plt
import time  # Dodano za mjerenje vremena

start_time = time.perf_counter()

img = cv2.imread('C:/Users/Safet/Desktop/cudaparalelno/input.jpg', cv2.IMREAD_GRAYSCALE)

if img is None:
    print("Greška: Slika nije učitana. Provjeri putanju!")
    exit()

sobel_x = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3)
sobel_y = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=3)
sobel_combined = cv2.magnitude(sobel_x, sobel_y)

end_time = time.perf_counter()

execution_time_ms = (end_time - start_time) * 1000
print(f"Vrijeme izvođenja: {execution_time_ms:.3f} ms")

plt.figure(figsize=(15,5))
plt.subplot(1,3,1), plt.imshow(img, cmap='gray'), plt.title('Original')
plt.subplot(1,3,2), plt.imshow(np.abs(sobel_x), cmap='gray'), plt.title('Sobel X')
plt.subplot(1,3,3), plt.imshow(np.abs(sobel_combined), cmap='gray'), plt.title('Kombinirano')
plt.show()
