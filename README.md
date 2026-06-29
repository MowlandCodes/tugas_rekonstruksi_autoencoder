# Tugas Autoencoder: Training di Kaggle dan Rekonstruksi Citra Fashion-MNIST melalui Decoder

Repositori ini berisi implementasi **Variational Autoencoder (VAE)** untuk melakukan kompresi *latent space* dan rekonstruksi citra dengan menggunakan dataset **Fashion-MNIST**. Proyek ini dikembangkan menggunakan manajemen runtime modern berbasis `uv` guna memastikan efisiensi lingkungan eksekusi (*environment*) serta proses resolusi dependensi yang cepat dan optimal.

## Identitas Mahasiswa
* **Nama:** M. Faridh Maulana
* **NIM:** 452024611065
* **Program Studi:** Teknik Informatika

---

## Lingkungan Eksekusi & Dependensi
Proyek ini dikelola dengan menggunakan **uv** (sebuah pengelola paket dan runtime Python yang cepat dan efisien).

* **Versi Python:** 3.14.x
* **Pustaka Utama (Core Libraries):**
  * torch (PyTorch Core)
  * torchvision (Transformasi dan pemrosesan citra)
  * numpy
  * pandas
  * matplotlib (Visualisasi grafik loss dan grid citra)
  * tqdm (Indikator progres pada antarmuka baris perintah)
  * easydict

### Prosedur Konfigurasi Lingkungan Menggunakan uv:
Eksekusi perintah berikut pada terminal sistem operasi Anda untuk menginisialisasi lingkungan virtual dan menginstal seluruh dependensi yang diperlukan:
```bash
# Install uv jika belum terpasang
# macOS and Linux.
curl -LsSf https://astral.sh/uv/install.sh | sh

# Windows.
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"

# Kemudian, jalankan perintah berikut untuk menginstal dependensi proyek:
uv sync
```

## Panduan Pelatihan (Training) di Kaggle
Proses pelatihan untuk mengekstrak bobot parameter (weights) model diisolasi di dalam **Kaggle Notebook** guna memanfaatkan akselerasi perangkat keras GPU NVIDIA.

1. Buka dasbor Kaggle Anda, buat notebook baru atau impor berkas `autoencoder-fashion-mnist.ipynb`.
2. Tambahkan dataset Fashion-MNIST dari Zalando Research ke dalam input notebook. Pastikan jalur (path) dataset mengarah ke direktori berikut:
  - `/kaggle/input/datasets/organizations/zalando-research/fashionmnist/fashion-mnist_train.csv`
  - `/kaggle/input/datasets/organizations/zalando-research/fashionmnist/fashion-mnist_test.csv`
3. Jalankan blok kode pada bagian *Entry Point* Otomatis untuk mengeksekusi proses pelatihan terhadap tiga variasi dimensi laten (*latent dimension*), yaitu 2, 8, dan 32 secara berurutan.
4. Setelah proses selesai, unduh berkas bobot model (berformat `.pth`) serta grafik riwayat loss hasil pelatihan dari direktori /kaggle/working/ ke penyimpanan lokal Anda.

## Panduan Inferensi & Rekonstruksi Melalui Terminal
Setelah mendapatkan berkas bobot model (`.pth`), proses inferensi wajib dijalankan secara mandiri melalui antarmuka baris perintah (Command Line Interface/CLI) pada terminal lokal (bukan melalui notebook).

Skrip `reconstruct.py` memuat arsitektur model secara utuh (Encoder dan Decoder), mengambil citra asli dari dataset berdasarkan parameter `--index`, kemudian menghasilkan luaran berupa citra rekonstruksi.

Contoh Perintah Eksekusi pada Terminal (Latent Dimension = 8, Indeks = 25):
```bash
uv run python rekonstruksi.py --model models/model_VAE_latent-8.pth --index 25 --latent_dim 8
```

## Daftar Berkas Keluaran (Output Files) yang Dihasilkan
Setelah seluruh skrip berhasil dieksekusi dengan sukses, direktori lokal Anda akan memuat beberapa berkas luaran sebagai berikut:

### Hasil Proses Pelatihan (Kaggle/Lokal):

- `model_VAE_latent-2.pth` : Berkas bobot model VAE dengan ruang laten berdimensi 2.
- `model_VAE_latent-8.pth` : Berkas bobot model VAE dengan ruang laten berdimensi 8.
- `model_VAE_latent-32.pth` : Berkas bobot model VAE dengan ruang laten berdimensi 32.
- `loss_history_dim*.png` : Grafik kurva penurunan nilai loss fungsional untuk masing-masing dimensi.

### Hasil Eksekusi Skrip `rekonstruksi.py`:
- `original.png` : Citra asli dari dataset Fashion-MNIST yang diekstrak berdasarkan indeks yang ditentukan.
- `reconstructed.png` : Citra sintetik hasil rekonstruksi ulang oleh lapisan Decoder.
- `comparison.png` : Lembar komparasi visual berdampingan (side-by-side) antara citra asli dan citra rekonstruksi sebagai bukti dokumentasi laporan.
