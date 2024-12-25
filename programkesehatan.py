# Program Self-Assessment Kesehatan Menggunakan Pohon Keputusan Sederhana
# Penulis: Kenneth Poenadi - 135230401

def input_yes_no(prompt):
    while True:
        response = input(prompt + " (Ya/Tidak): ").strip().lower()
        if response in ['ya', 'tidak']:
            return response
        else:
            print("Input tidak valid. Silakan masukkan 'Ya' atau 'Tidak'.")

def input_integer(prompt, min_value=None, max_value=None):
    while True:
        try:
            value = int(input(prompt + ": ").strip())
            if (min_value is not None and value < min_value) or \
               (max_value is not None and value > max_value):
                if min_value is not None and max_value is not None:
                    print(f"Silakan masukkan angka antara {min_value} dan {max_value}.")
                elif min_value is not None:
                    print(f"Silakan masukkan angka minimal {min_value}.")
                elif max_value is not None:
                    print(f"Silakan masukkan angka maksimal {max_value}.")
                continue
            return value
        except ValueError:
            print("Input tidak valid. Silakan masukkan angka yang benar.")

def input_blood_pressure(prompt):
    while True:
        bp = input(prompt + " (format: sistolik/diastolik, misalnya 120/80): ").strip()
        if '/' in bp:
            try:
                sistolik, diastolik = map(int, bp.split('/'))
                if sistolik <= 0 or diastolik <= 0:
                    print("Angka tekanan darah harus positif.")
                    continue
                return sistolik, diastolik
            except ValueError:
                print("Format salah. Silakan masukkan dalam format sistolik/diastolik.")
        else:
            print("Format salah. Silakan masukkan dalam format sistolik/diastolik.")

def main():
    print("=== Program Self-Assessment Kesehatan Menggunakan Pohon Keputusan Sederhana ===\n")

    # Input Data Pengguna
    nama = input("Masukkan Nama Anda: ").strip()
    usia = input_integer("Masukkan Usia Anda", min_value=1, max_value=120)
    jenis_kelamin = input("Masukkan Jenis Kelamin Anda (Laki-laki/Perempuan): ").strip().lower()
    while jenis_kelamin not in ['laki-laki', 'perempuan']:
        print("Input tidak valid. Silakan masukkan 'Laki-laki' atau 'Perempuan'.")
        jenis_kelamin = input("Masukkan Jenis Kelamin Anda (Laki-laki/Perempuan): ").strip().lower()

    riwayat_keluarga = input_yes_no("Apakah Anda memiliki riwayat kesehatan keluarga?")
    merokok = input_yes_no("Apakah Anda merokok?")

    # Tekanan Darah
    sistolik, diastolik = input_blood_pressure("Masukkan Tekanan Darah Anda")

    # Kolesterol
    kolesterol = input_integer("Masukkan Tingkat Kolesterol Anda (mg/dL)", min_value=100, max_value=400)

    # Frekuensi Batuk
    print("\nFrekuensi Batuk:")
    print("1. Rendah")
    print("2. Sedang")
    print("3. Tinggi")
    frekuensi_batuk_input = input_integer("Pilih frekuensi batuk Anda (1/2/3)", min_value=1, max_value=3)
    frekuensi_map = {1: 'Rendah', 2: 'Sedang', 3: 'Tinggi'}
    frekuensi_batuk = frekuensi_map.get(frekuensi_batuk_input, 'Rendah')

    demam = input_yes_no("Apakah Anda mengalami demam?")

    # Durasi Gejala
    durasi = input_integer("Masukkan Durasi Gejala Anda (dalam hari)", min_value=1, max_value=365)

    # Mengumpulkan data pengguna
    user_data = {
        'nama': nama,
        'usia': usia,
        'jenis_kelamin': jenis_kelamin,
        'riwayat_keluarga': riwayat_keluarga,
        'merokok': merokok,
        'sistolik': sistolik,
        'diastolik': diastolik,
        'kolesterol': kolesterol,
        'frekuensi_batuk': frekuensi_batuk,
        'demam': demam,
        'durasi': durasi
    }

    # Menampilkan Informasi Pengguna
    print("\n=== Hasil Diagnosis Awal ===")
    print(f"Nama: {nama}")
    print(f"Usia: {usia} Tahun")
    print(f"Jenis Kelamin: {jenis_kelamin.capitalize()}")
    print(f"Riwayat Kesehatan Keluarga: {riwayat_keluarga.capitalize()}")
    print(f"Merokok: {merokok.capitalize()}")
    print(f"Tekanan Darah: {sistolik}/{diastolik} mmHg")
    print(f"Kolesterol: {kolesterol} mg/dL")
    print(f"Frekuensi Batuk: {frekuensi_batuk}")
    print(f"Demam: {demam.capitalize()}")
    print(f"Durasi Gejala: {durasi} Hari")

    # Logika Diagnosis dan Rekomendasi
    diagnosis = []
    rekomendasi = []

    # Cek Hipertensi
    hipertensi = False
    if sistolik >= 140 or diastolik >= 90:
        hipertensi = True

    # Cek Kolesterol Tinggi
    kolesterol_tinggi = False
    if kolesterol > 200:
        kolesterol_tinggi = True
    elif 180 < kolesterol <= 200:
        kolesterol_tinggi = True  # Sesuaikan dengan aturan yang diinginkan

    # Analisis Hipertensi dan Kolesterol
    if hipertensi:
        if kolesterol_tinggi:
            if riwayat_keluarga == 'ya':
                diagnosis.append("Risiko Penyakit Kardiovaskular Tinggi")
                rekomendasi.append("Segera konsultasi dengan dokter untuk evaluasi lebih lanjut.")
            else:
                diagnosis.append("Pemantauan dan Pengelolaan Kolesterol Diperlukan")
                rekomendasi.append("Perubahan gaya hidup dan diet sehat dianjurkan.")
        else:
            diagnosis.append("Pemantauan Tekanan Darah Diperlukan")
            rekomendasi.append("Lakukan pemeriksaan rutin dan pertimbangkan perubahan gaya hidup.")
    else:
        diagnosis.append("Tekanan Darah Stabil")

    # Analisis Batuk
    if frekuensi_batuk in ['Sedang', 'Tinggi']:
        if demam == 'ya':
            diagnosis.append("Kemungkinan Infeksi Virus atau Bakteri")
            rekomendasi.append("Segera konsultasi dengan dokter untuk diagnosis dan pengobatan.")
        else:
            diagnosis.append("Kemungkinan Batuk Non-Infeksi (Alergi, Asma, Iritasi)")
            if durasi > 7:
                rekomendasi.append("Segera konsultasi dengan dokter untuk evaluasi lebih lanjut.")
            else:
                rekomendasi.append("Monitor gejala dan perhatikan jika memburuk.")
    else:
        diagnosis.append("Tidak Ada Gejala Batuk yang Signifikan")

    # Rekomendasi Berdasarkan Durasi
    if durasi > 7:
        rekomendasi.append("Durasi gejala lebih dari 7 hari. Segera konsultasi ke dokter.")
    else:
        rekomendasi.append("Durasi gejala masih dalam batas normal. Monitor gejala Anda.")

    # Tambahkan rekomendasi berhenti merokok
    if merokok == 'ya':
        rekomendasi.append("Rekomendasi: Berhenti merokok untuk mengurangi risiko penyakit kardiovaskular dan kesehatan lainnya.")

    # Output Diagnosis
    print("\n**Kemungkinan Diagnosis:**")
    for d in diagnosis:
        print(f"- {d}")

    print("\n**Rekomendasi:**")
    for r in rekomendasi:
        print(f"- {r}")

if __name__ == "__main__":
    main()
