import time
import cv2
import numpy as np
import urllib.request
import matplotlib.pyplot as plt

url = 'https://upload.wikimedia.org/wikipedia/commons/thumb/c/cd/Chinstrap_Penguin.jpg/2336px-Chinstrap_Penguin.jpg'

resp = urllib.request.urlopen(url)
image_array = np.asarray(bytearray(resp.read()), dtype=np.uint8)
img = cv2.imdecode(image_array, cv2.IMREAD_GRAYSCALE)

if img is None:
    raise ValueError("Slika nije učitana s URL-a.")

start_time = time.time()

sobel_x = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3)
sobel_y = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=3)
sobel_combined = cv2.magnitude(sobel_x, sobel_y)

end_time = time.time()

print(f"Vrijeme obrade Sobel filtera: {end_time - start_time:.5f} sekundi")

plt.figure(figsize=(15, 5))
plt.subplot(1, 3, 1), plt.imshow(img, cmap='gray'), plt.title('Original')
plt.subplot(1, 3, 2), plt.imshow(np.abs(sobel_x), cmap='gray'), plt.title('Sobel X')
plt.subplot(1, 3, 3), plt.imshow(np.abs(sobel_combined), cmap='gray'), plt.title('Kombinirano')
plt.tight_layout()
plt.show()
