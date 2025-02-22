import imageio.v3 as imageio
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import sobel, convolve

def roberts_operator(image):
    roberts_x = np.array([[1, 0], [0, -1]])
    roberts_y = np.array([[0, 1], [-1, 0]])
    
    edge_x = convolve(image, roberts_x)
    edge_y = convolve(image, roberts_y)
    
    return np.sqrt(edge_x**2 + edge_y**2)

def main():
    # Baca gambar berwarna
    image_color = imageio.imread('D:/hewan.jpeg')  # Ganti dengan path gambar Anda
    
    # Konversi ke grayscale
    image_gray = np.dot(image_color[..., :3], [0.2989, 0.5870, 0.1140])
    
    # Deteksi tepi dengan Roberts
    edge_roberts = roberts_operator(image_gray)
    
    # Deteksi tepi dengan Sobel
    edge_sobel = np.hypot(sobel(image_gray, axis=0), sobel(image_gray, axis=1))
    
    # Buat tata letak gambar: 2 baris x 3 kolom
    fig, axes = plt.subplots(2, 3, figsize=(12, 10), gridspec_kw={'height_ratios': [1, 1]})

    # Gambar grayscale (tengah atas)
    axes[0, 1].imshow(image_gray, cmap='gray')
    axes[0, 1].set_title('Gambar Grayscale (Asli)')
    axes[0, 1].axis('off')

    # Kosongkan subplot kiri atas dan kanan atas agar gambar grayscale berada di tengah
    fig.delaxes(axes[0, 0])
    fig.delaxes(axes[0, 2])

    # Deteksi tepi - Roberts (kiri bawah)
    axes[1, 0].imshow(edge_roberts, cmap='gray')
    axes[1, 0].set_title('Deteksi Tepi - Roberts')
    axes[1, 0].axis('off')

    # Deteksi tepi - Sobel (kanan bawah)
    axes[1, 2].imshow(edge_sobel, cmap='gray')
    axes[1, 2].set_title('Deteksi Tepi - Sobel')
    axes[1, 2].axis('off')

    # Analisis Roberts (di bawah gambar Roberts)
    axes[1, 1].text(0, 1, 
                    "Analisis Roberts:\n"
                    "- Lebih sensitif terhadap perubahan kecil.\n"
                    "- Menghasilkan tepi yang lebih kasar.\n"
                    "- Cocok untuk gambar dengan kontras tinggi.\n\n"
                    "Analisis Sobel:\n"
                    "- Lebih halus dan stabil terhadap noise.\n"
                    "- Menangkap lebih banyak detail dalam struktur gambar.\n"
                    "- Memberikan transisi tepi yang lebih baik.\n",
                    fontsize=12, ha='left', va='top')
    axes[1, 1].axis('off')

    # Tambahkan kesimpulan perbandingan di bawahnya
    fig.text(0.5, 0.05, 
             "Perbandingan: \n" "Metode Roberts lebih sensitif terhadap perubahan kecil dan cocok untuk gambar dengan kontras tinggi,\n "
             "tetapi kurang tahan terhadap noise. Metode Sobel memberikan hasil lebih halus, stabil terhadap noise, \n"
             "dan lebih baik dalam menangkap detail pada gambar kompleks.\n",
             ha='center', fontsize=12)
    

    # Sesuaikan tata letak agar tidak bertabrakan
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.15)
    plt.show()

if __name__ == "__main__":
    main()
