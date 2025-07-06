import streamlit as st
import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage.feature import graycomatrix, graycoprops
from scipy.stats import skew, kurtosis
import io

# ==============================================================================
# --- BAGIAN 1: FUNGSI-FUNGSI PENGOLAHAN CITRA DASAR ---
# ==============================================================================

@st.cache_data
def convert_image_to_grayscale(image):
    """Konversi citra berwarna ke grayscale."""
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

@st.cache_data
def calculate_histogram(image):
    """Hitung histogram citra."""
    hist = cv2.calcHist([image], [0], None, [256], [0, 256])
    hist_normalized = hist / hist.sum()
    return hist, hist_normalized

@st.cache_data
def equalize_histogram(image):
    """Lakukan equalization histogram."""
    return cv2.equalizeHist(image)


# ==============================================================================
# --- BAGIAN 2: FUNGSI-FUNGSI EKSTRAKSI FITUR ---
# ==============================================================================

@st.cache_data
def find_contours_and_signature(grayscale_image):
    """Mencari kontur dan menghitung tanda tangan kontur."""
    _, thresh = cv2.threshold(grayscale_image, 127, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    if contours:
        largest_contour = max(contours, key=cv2.contourArea)
        M = cv2.moments(largest_contour)
        if M["m00"] != 0:
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
            contour_signature = [np.linalg.norm(np.array([p[0][0], p[0][1]]) - np.array([cx, cy])) for p in largest_contour]
            return largest_contour, contour_signature, (cx, cy)
    return None, None, None

@st.cache_data
def calculate_glcm_features(grayscale_image):
    """Hitung fitur tekstur GLCM."""
    glcm_image = (grayscale_image / 255 * (256 - 1)).astype(int)
    distances = [1]
    angles = [0, np.pi/4, np.pi/2, 3*np.pi/4]
    levels = 256

    if glcm_image.shape[0] < np.max(distances) + 1 or glcm_image.shape[1] < np.max(distances) + 1:
        return "Citra terlalu kecil untuk menghitung GLCM dengan jarak yang diberikan."

    glcm = graycomatrix(glcm_image, distances=distances, angles=angles, levels=levels, symmetric=True, normed=True)
    properties = ['contrast', 'dissimilarity', 'homogeneity', 'energy', 'correlation', 'ASM']
    feature_results = {}
    for prop in properties:
        feature_value = graycoprops(glcm, prop)
        feature_results[prop] = feature_value.mean()
    return feature_results

@st.cache_data
def calculate_color_statistics(image):
    """Hitung statistik warna (mean, variance, skewness, kurtosis)."""
    b_channel, g_channel, r_channel = cv2.split(image)
    channels = {'Biru': b_channel, 'Hijau': g_channel, 'Merah': r_channel}
    stats = {}
    for name, channel in channels.items():
        stats[name] = {
            'Mean': np.mean(channel),
            'Variance': np.var(channel),
            'Skewness': skew(channel.flatten()),
            'Kurtosis': kurtosis(channel.flatten())
        }
    return stats


# ==============================================================================
# --- BAGIAN 3: FUNGSI-FUNGSI OPERASI MORFOLOGI ---
# ==============================================================================

@st.cache_data
def perform_morphological_operations(binary_image, kernel_size=5):
    """Melakukan operasi morfologi."""
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    dilated = cv2.dilate(binary_image, kernel, iterations=1)
    eroded = cv2.erode(binary_image, kernel, iterations=1)
    opened = cv2.morphologyEx(binary_image, cv2.MORPH_OPEN, kernel)
    closed = cv2.morphologyEx(binary_image, cv2.MORPH_CLOSE, kernel)
    return dilated, eroded, opened, closed


# ==============================================================================
# --- BAGIAN 4: FUNGSI-FUNGSI PENGOLAHAN CITRA BERWARNA ---
# ==============================================================================

@st.cache_data
def transform_color_spaces(image):
    """Transformasi ruang warna."""
    # Menggunakan HSV (Hue, Saturation, Value) karena ini adalah fungsi bawaan OpenCV
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    # YCbCr menggunakan OpenCV
    ycbcr_image = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb) 
    
    return hsv_image, ycbcr_image # Mengembalikan citra HSV dan citra YCbCr


# ==============================================================================
# --- BAGIAN 5: FUNGSI-FUNGSI ANAGLYPH ---
# ==============================================================================

@st.cache_data
def apply_anaglyph(image, shift_amount=5):
    """Implementasi Anaglyph sederhana (Merah-Cyan) dari satu citra."""
    # Pastikan gambar adalah BGR
    if image.shape[2] == 4: # Jika ada alpha channel, konversi ke BGR
        image = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)

    # Pisahkan saluran warna (OpenCV: B, G, R)
    b, g, r = cv2.split(image)

    # Buat saluran kosong untuk gambar anaglyph
    anaglyph_r = r.copy() # Saluran Merah untuk mata kiri (dari gambar asli)
    anaglyph_g = np.zeros_like(g) # Saluran Hijau untuk mata kanan (digeser)
    anaglyph_b = np.zeros_like(b) # Saluran Biru untuk mata kanan (digeser)

    # Geser saluran Hijau dan Biru ke kanan
    anaglyph_g[:, shift_amount:] = g[:, :-shift_amount]
    anaglyph_b[:, shift_amount:] = b[:, :-shift_amount]

    # Gabungkan saluran untuk membentuk gambar anaglyph
    # Urutan: B, G, R untuk OpenCV
    anaglyph_image = cv2.merge([anaglyph_b, anaglyph_g, anaglyph_r])

    return anaglyph_image


# ==============================================================================
# --- BAGIAN 6: FUNGSI-FUNGSI STEGANOGRAFI DAN WATERMARKING ---
# ==============================================================================

@st.cache_data
def hide_message_lsb(image, message):
    """Menyembunyikan pesan teks dalam gambar menggunakan LSB."""
    binary_message = ''.join(format(ord(char), '08b') for char in message)
    binary_message += '1111111111111110' # Delimiter

    img_copy = image.copy()
    flat_img = img_copy.flatten()

    if len(binary_message) > len(flat_img):
        return None

    for i in range(len(binary_message)):
        pixel_value = flat_img[i]
        pixel_value = pixel_value & 0xFE
        pixel_value = pixel_value | int(binary_message[i])
        flat_img[i] = pixel_value

    return flat_img.reshape(img_copy.shape)

@st.cache_data
def reveal_message_lsb(image):
    """Mengungkap pesan teks dari gambar menggunakan LSB."""
    binary_message = ""
    flat_img = image.flatten()
    delimiter = '1111111111111110'
    delim_len = len(delimiter)

    for i in range(len(flat_img)):
        binary_message += str(flat_img[i] & 1)
        if len(binary_message) >= delim_len and binary_message[-delim_len:] == delimiter:
            break
    
    binary_message = binary_message[:-delim_len]

    if not binary_message:
        return "Tidak ada pesan tersembunyi atau delimiter tidak ditemukan."

    message = ""
    for i in range(0, len(binary_message), 8):
        byte = binary_message[i:i+8]
        if len(byte) == 8:
            try:
                message += chr(int(byte, 2))
            except ValueError: # Tangani jika byte bukan ASCII valid
                message += "?"
        
    return message


@st.cache_data
def apply_watermark(image, logo_file=None):
    """Menyisipkan watermark pada citra."""
    watermarked_image = image.copy()

    logo = None
    if logo_file is not None:
        file_bytes = np.asarray(bytearray(logo_file.read()), dtype=np.uint8)
        logo = cv2.imdecode(file_bytes, cv2.IMREAD_UNCHANGED)
        if logo is not None:
            # Resize logo to 1/10th height of the original image
            logo_height = image.shape[0] // 10
            logo_width = int(logo.shape[1] * (logo_height / logo.shape[0]))
            logo = cv2.resize(logo, (logo_width, logo_height))
        else:
            st.warning("Tidak dapat membaca file logo yang diunggah. Menggunakan logo default.")

    if logo is None:
        # Create a simple white square logo if no file or failed to load
        logo_size = image.shape[0] // 10
        logo = np.ones((logo_size, logo_size, 3), dtype=np.uint8) * 255

    x_offset = watermarked_image.shape[1] - logo.shape[1] - 10
    y_offset = watermarked_image.shape[0] - logo.shape[0] - 10

    x_offset = max(0, x_offset)
    y_offset = max(0, y_offset)

    # Handle alpha channel for transparency
    if logo.shape[2] == 4:
        alpha_s = logo[:, :, 3] / 255.0
        alpha_l = 1.0 - alpha_s
        for c in range(0, 3):
            watermarked_image[y_offset:y_offset+logo.shape[0], x_offset:x_offset+logo.shape[1], c] = \
                (alpha_s * logo[:, :, c] + alpha_l * watermarked_image[y_offset:y_offset+logo.shape[0], x_offset:x_offset+logo.shape[1], c])
    else:
        watermarked_image[y_offset:y_offset+logo.shape[0], x_offset:x_offset+logo.shape[1]] = logo

    return watermarked_image


# ==============================================================================
# --- BAGIAN UTAMA: STREAMLIT UI DAN ALUR PROSES ---
# ==============================================================================

st.set_page_config(layout="wide", page_title="UAS Pengolahan Citra Digital")

st.title("UAS Pengolahan Citra Digital")
st.markdown("Aplikasi interaktif untuk mengaplikasikan konsep dan teknik pengolahan citra digital.")

st.sidebar.header("Pengaturan Input")
uploaded_file = st.sidebar.file_uploader("1. Unggah Pas Foto Anda (wajib, min. 512x512 piksel)", type=["jpg", "jpeg", "png"])
uploaded_logo = st.sidebar.file_uploader("2. Unggah Logo untuk Watermarking (wajib)", type=["jpg", "jpeg", "png"])
secret_message_input = st.sidebar.text_input("3. Pesan Rahasia untuk Steganografi (wajib)")

# Tombol untuk memulai proses
process_button = st.sidebar.button("Mulai Proses Pengolahan Citra")

# Bagian utama akan diisi hanya setelah tombol diklik dan semua input valid
if process_button:
    # Cek apakah semua input sudah terisi
    if uploaded_file is None:
        st.error("❗ Harap unggah Pas Foto Anda terlebih dahulu.")
    elif uploaded_logo is None:
        st.error("❗ Harap unggah Logo untuk Watermarking terlebih dahulu.")
    elif not secret_message_input.strip(): # Memastikan pesan tidak kosong atau hanya spasi
        st.error("❗ Harap masukkan Pesan Rahasia untuk Steganografi.")
    else:
        # Semua input valid, lanjutkan proses
        st.success("✅ Semua input terisi. Memulai proses pengolahan citra...")
        
        # Konversi file yang diunggah ke array numpy OpenCV
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        original_image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

        if original_image is None:
            st.error("Gagal memuat gambar. Pastikan ini adalah file gambar yang valid.")
        else:
            # --- 1. Pengenalan Citra ---
            st.subheader("1. Pengenalan Citra")
            st.write(f"Ukuran citra asli: {original_image.shape[1]}x{original_image.shape[0]} piksel")
            st.image(cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB), caption="Citra Asli", use_container_width=True)

            # 1.1 Konversi ke Grayscale
            grayscale_image = convert_image_to_grayscale(original_image)
            st.image(grayscale_image, caption="Citra Grayscale", use_container_width=True, channels="GRAY")

            # 1.2 Histogram dan Normalisasi
            hist_orig, hist_norm = calculate_histogram(grayscale_image)
            
            fig1, ax1 = plt.subplots()
            ax1.plot(hist_orig)
            ax1.set_title("Histogram Grayscale Original")
            ax1.set_xlabel("Intensitas Piksel")
            ax1.set_ylabel("Frekuensi")
            st.pyplot(fig1)

            fig2, ax2 = plt.subplots()
            ax2.plot(hist_norm)
            ax2.set_title("Histogram Grayscale Normalisasi")
            ax2.set_xlabel("Intensitas Piksel")
            ax2.set_ylabel("Frekuensi")
            st.pyplot(fig2)

            # 1.3 Equalization Histogram
            equalized_image = equalize_histogram(grayscale_image)
            hist_equalized, _ = calculate_histogram(equalized_image)
            st.image(equalized_image, caption="Citra Grayscale Hasil Equalization", use_container_width=True, channels="GRAY")

            fig3, ax3 = plt.subplots()
            ax3.plot(hist_equalized)
            ax3.set_title("Histogram Grayscale Hasil Equalization")
            ax3.set_xlabel("Intensitas Piksel")
            ax3.set_ylabel("Frekuensi")
            st.pyplot(fig3)

            # --- 2. Ekstraksi Fitur ---
            st.subheader("2. Ekstraksi Fitur")

            # 2.1 Kontur dan Tanda Tangan Kontur
            st.markdown("### Ekstraksi Fitur Kontur (Tanda Tangan Kontur)")
            largest_contour, contour_signature, centroid = find_contours_and_signature(grayscale_image)

            if largest_contour is not None:
                contour_display_image = np.zeros_like(original_image)
                cv2.drawContours(contour_display_image, [largest_contour], -1, (0, 255, 0), 2)
                st.image(cv2.cvtColor(contour_display_image, cv2.COLOR_BGR2RGB), caption="Kontur Terbesar yang Diekstrak", use_container_width=True)

                if contour_signature is not None:
                    fig4, ax4 = plt.subplots()
                    ax4.plot(contour_signature)
                    ax4.set_title("Tanda Tangan Kontur (Jarak dari Centroid)")
                    ax4.set_xlabel("Indeks Titik Kontur")
                    ax4.set_ylabel("Jarak")
                    st.pyplot(fig4)
                else:
                    st.warning("Tidak dapat menghitung tanda tangan kontur.")
            else:
                st.warning("Tidak ada kontur yang ditemukan pada citra.")

            # 2.2 Fitur Tekstur GLCM
            st.markdown("### Ekstraksi Fitur Tekstur (GLCM)")
            glcm_features = calculate_glcm_features(grayscale_image)
            if isinstance(glcm_features, str): # Error message
                st.error(glcm_features)
            else:
                st.write("Fitur Tekstur GLCM:")
                for prop, value in glcm_features.items():
                    st.write(f"- **{prop.capitalize()}**: {value:.4f}")

            # 2.3 Statistik Warna
            st.markdown("### Statistik Warna Citra Asli")
            color_stats = calculate_color_statistics(original_image)
            for channel, stats_data in color_stats.items():
                st.write(f"**--- Saluran {channel} ---**")
                for stat_name, stat_value in stats_data.items():
                    st.write(f"  - {stat_name}: {stat_value:.4f}")

            # --- 3. Implementasi Operasi Morfologi ---
            st.subheader("3. Implementasi Operasi Morfologi")
            _, binary_image = cv2.threshold(grayscale_image, 127, 255, cv2.THRESH_BINARY)
            st.image(binary_image, caption="Citra Biner (Hasil Thresholding)", use_container_width=True, channels="GRAY")

            dilated_image, eroded_image, opened_image, closed_image = perform_morphological_operations(binary_image)

            col1, col2 = st.columns(2)
            with col1:
                st.image(dilated_image, caption="Citra Hasil Dilasi", use_container_width=True, channels="GRAY")
            with col2:
                st.image(eroded_image, caption="Citra Hasil Erosi", use_container_width=True, channels="GRAY")

            col3, col4 = st.columns(2)
            with col3:
                st.image(opened_image, caption="Citra Hasil Opening", use_container_width=True, channels="GRAY")
            with col4:
                st.image(closed_image, caption="Citra Hasil Closing", use_container_width=True, channels="GRAY")

            # --- 4. Pengolahan Citra Berwarna ---
            st.subheader("4. Pengolahan Citra Berwarna")
            # Panggil fungsi transform_color_spaces
            hsv_image, ycbcr_image = transform_color_spaces(original_image)

            st.markdown("### Citra Hasil Transformasi HSV (Hue, Saturation, Value)")
            h, s, v = cv2.split(hsv_image)
            col_hsv1, col_hsv2, col_hsv3 = st.columns(3)
            with col_hsv1:
                st.image(h, caption="Saluran Hue (H)", use_container_width=True, channels="GRAY")
            with col_hsv2:
                st.image(s, caption="Saluran Saturation (S)", use_container_width=True, channels="GRAY")
            with col_hsv3:
                st.image(v, caption="Saluran Value (V)", use_container_width=True, channels="GRAY")
            st.markdown("""
            * **Hue (H):** Merepresentasikan jenis warna (misalnya, merah, hijau, biru). Nilai piksel menunjukkan posisi warna pada roda warna.
            * **Saturation (S):** Merepresentasikan kemurnian warna atau seberapa jauh warna dari abu-abu. Nilai tinggi berarti warna lebih murni/cerah, nilai rendah berarti lebih pudar/mendekati abu-abu.
            * **Value (V):** Merepresentasikan kecerahan atau intensitas warna. Nilai tinggi berarti lebih terang, nilai rendah berarti lebih gelap.
            """)
            
            st.markdown("### Citra Hasil Transformasi YCbCr (Luminance, Chroma Blue, Chroma Red)")
            y, cb, cr = cv2.split(ycbcr_image)
            col_ycbcr1, col_ycbcr2, col_ycbcr3 = st.columns(3)
            with col_ycbcr1:
                st.image(y, caption="Saluran Luminance (Y)", use_container_width=True, channels="GRAY")
            with col_ycbcr2:
                st.image(cb, caption="Saluran Chroma Blue (Cb)", use_container_width=True, channels="GRAY")
            with col_ycbcr3:
                st.image(cr, caption="Saluran Chroma Red (Cr)", use_container_width=True, channels="GRAY")
            st.markdown("""
            * **Luminance (Y):** Merepresentasikan informasi kecerahan atau intensitas (mirip citra grayscale). Ini adalah komponen yang paling penting untuk persepsi mata manusia.
            * **Chroma Blue (Cb):** Merepresentasikan perbedaan antara komponen biru dan *luminance*. Mengandung informasi warna biru-kuning.
            * **Chroma Red (Cr):** Merepresentasikan perbedaan antara komponen merah dan *luminance*. Mengandung informasi warna merah-hijau.
            """)

            # Opsional: Tampilkan juga gambar yang dikonversi kembali ke RGB untuk perbandingan visual
            st.markdown("### Visualisasi Kembali ke RGB (untuk perbandingan)")
            col_reconv1, col_reconv2 = st.columns(2)
            with col_reconv1:
                st.image(cv2.cvtColor(hsv_image, cv2.COLOR_HSV2RGB), caption="HSV Dikonversi Kembali ke RGB", use_container_width=True)
            with col_reconv2:
                st.image(cv2.cvtColor(ycbcr_image, cv2.COLOR_YCrCb2RGB), caption="YCbCr Dikonversi Kembali ke RGB", use_container_width=True)
            st.markdown("""
            * **Catatan:** Gambar-gambar di atas adalah hasil transformasi ruang warna yang kemudian dikonversi kembali ke format RGB untuk tujuan visualisasi. Meskipun secara internal data warna telah diorganisir ulang (misalnya, menjadi komponen H, S, V atau Y, Cb, Cr), visualisasi akhir dalam RGB mungkin terlihat mirip dengan citra asli jika tidak ada manipulasi yang dilakukan pada saluran individual di ruang warna baru. Perbedaan utama terletak pada bagaimana informasi warna dan kecerahan dipisahkan, yang sangat berguna untuk aplikasi pengolahan citra tertentu (misalnya, penyesuaian warna, kompresi, segmentasi).
            """)
            
            # --- 5. Bagian Anaglyph ---
            st.subheader("5. Anaglyph")
            st.markdown("### Implementasi Anaglyph")
            anaglyph_image = apply_anaglyph(original_image)
            st.image(cv2.cvtColor(anaglyph_image, cv2.COLOR_BGR2RGB), caption="Citra Hasil Anaglyph", use_container_width=True)
            st.markdown("""
            **Proses Anaglyph Sederhana:**
            * Anaglyph ini dibuat dengan memisahkan saluran warna Merah, Hijau, dan Biru dari citra asli.
            * Saluran Merah dari citra asli digunakan untuk mata kiri.
            * Saluran Hijau dan Biru digeser sedikit secara horizontal (ke kanan dalam kasus ini) untuk menciptakan perspektif yang berbeda untuk mata kanan.
            * Ketiga saluran yang telah dimodifikasi ini kemudian digabungkan kembali.
            * Ketika dilihat dengan kacamata Anaglyph (merah-cyan), setiap lensa menyaring warna tertentu, sehingga setiap mata hanya melihat satu perspektif, menciptakan ilusi kedalaman 3D.
            """)

            # --- 6. Steganografi dan Watermarking ---
            st.subheader("6. Steganografi dan Watermarking") # Nomor subheader diubah menjadi 6

            # 6.1 Steganografi LSB
            st.markdown("### Steganografi LSB")
            stego_image = hide_message_lsb(original_image, secret_message_input)
            if stego_image is not None:
                st.image(cv2.cvtColor(stego_image, cv2.COLOR_BGR2RGB), caption="Citra Hasil Steganografi LSB", use_container_width=True)
                st.success(f"Pesan tersembunyi: '{secret_message_input}'")
                # Verifikasi pesan
                revealed_message = reveal_message_lsb(stego_image)
                st.info(f"Pesan yang berhasil diungkap dari citra: '{revealed_message}'")
                st.markdown("""
                **Proses Steganografi LSB:**
                * Mengubah bit paling tidak signifikan (Least Significant Bit - LSB) dari setiap piksel citra.
                * Karena LSB memiliki dampak paling kecil pada nilai piksel, perubahan visual pada citra sangat minim atau tidak terlihat oleh mata manusia.
                * Pesan teks dikonversi menjadi urutan bit dan disisipkan satu per satu ke LSB setiap komponen warna piksel.
                """)
            else:
                st.error("Pesan terlalu panjang untuk disembunyikan dalam gambar ini.")

            # 6.2 Watermarking Sederhana
            st.markdown("### Watermarking Sederhana")
            watermarked_image = apply_watermark(original_image, uploaded_logo)
            st.image(cv2.cvtColor(watermarked_image, cv2.COLOR_BGR2RGB), caption="Citra Hasil Watermarking", use_container_width=True)
            st.markdown("""
            **Proses Watermarking Sederhana:**
            * Menyisipkan logo kecil di sudut kanan bawah citra asli.
            * Logo di-*resize* agar sesuai dan kemudian di-*overlay* ke area target pada citra.
            * Watermarking umumnya terlihat dan digunakan untuk menunjukkan kepemilikan atau otentikasi.
            """)

else: # Hanya tampilkan pesan ini jika tombol belum diklik atau pertama kali dibuka
    st.info("Silakan lengkapi semua input di sidebar kiri (pas foto, logo, pesan rahasia) lalu klik 'Mulai Proses Pengolahan Citra'.")

st.sidebar.markdown("---")
st.sidebar.markdown("Dibuat untuk Tugas UAS Pengolahan Citra Digital")
