import cv2
import numpy as np
import matplotlib.pyplot as plt

# Baca gambar
image_path = "D:/hewan.jpeg"
image = cv2.imread(image_path)

if image is None:
    print("Gambar tidak bisa dibuka! Periksa path dan format gambar.")
else:
    # Konversi ke RGB
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Ubah gambar menjadi array 2D (reshape ke bentuk (jumlah_pixel, 3))
    pixel_values = image.reshape((-1, 3))
    pixel_values = np.float32(pixel_values)  # Konversi ke tipe data float

    # Tentukan jumlah klaster (K)
    K = 3 

    # Kriteria stopping (maksimal iterasi 100, perubahan pusat klaster < 1.0)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 1.0)

    # Inisialisasi K-Means
    _, labels, centers = cv2.kmeans(pixel_values, K, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)

    # Konversi pusat klaster ke tipe integer
    centers = np.uint8(centers)

    # Ganti nilai piksel dengan warna klasternya
    segmented_image = centers[labels.flatten()]
    segmented_image = segmented_image.reshape(image.shape)

    # Tampilkan hasil
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(image)
    plt.title("Gambar Asli")
    plt.axis("off")

    plt.subplot(1, 2, 2)
    plt.imshow(segmented_image)
    plt.title(f"Segmentasi Citra (K={K})")
    plt.axis("off")

    plt.show()
