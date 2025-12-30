# Asisten Navigasi: Sistem Deteksi Objek Real-Time untuk Tunanetra Berbasis YOLOv11

 **Asisten Navigasi** adalah aplikasi seluler berbasis Android yang dirancang untuk membantu penyandang tunanetra dalam mengenali lingkungan sekitar secara mandiri. Aplikasi ini memanfaatkan kecerdasan buatan (AI) mutakhir untuk mendeteksi objek, mengestimasi jarak, dan memberikan umpan balik audio serta haptik secara real-time. Aplikasi ini juga mempertimbangkan spesifikasi dari seluruh pengguna sehingga dapat digunakan pada device low-end sekalipun. 

## 📸 Antarmuka Aplikasi

Berikut adalah tampilan antarmuka aplikasi yang dikembangkan:

<div align="center">
  
| **Onboarding & Fitur** | **Umpan Balik Haptik** |
|:---:|:---:|
| <img src="https://github.com/user-attachments/assets/16e9b773-134e-446d-a0f9-4d659264c5ed" width="600" alt="Halaman Selamat Datang"/> <br> *Halaman Sambutan* | <img src="https://github.com/user-attachments/assets/37be8cf2-6a60-46d7-96e7-4f5f59999a3a" width="600" alt="Getaran Haptik"/> <br> *Feedback Getaran* |

</div>

<div align="center">

| **Deteksi Real-Time** |
|:---:|
| <img src="https://github.com/user-attachments/assets/3cea5c81-cc6d-4699-8ca9-e734838dc172" width="900" alt="Deteksi Pintu dan Kursi"/> <br> *Deteksi Objek & Estimasi Jarak* |

</div>

<div align="center">

| **Fitur Deteksi** | **Panduan Suara** |
|:---:|:---:|
| <img src="https://github.com/user-attachments/assets/9a954ff0-1e4e-4b09-bcf8-6c593b71386a" width="600" alt="List Objek"/> <br> *5 Kelas Objek Utama* | <img src="https://github.com/user-attachments/assets/f2466e86-29b8-401f-ba23-95f69346d85c" width="600" alt="Panduan Suara"/> <br> *Notifikasi Audio* |

</div>
<br>



## ✨ Fitur Utama

Berdasarkan implementasi saat ini, aplikasi memiliki fitur unggulan:

1.  **Deteksi Objek Real-Time:** Menggunakan kamera ponsel untuk memindai lingkungan secara terus-menerus.
2.  **5 Kelas Objek Spesifik:** Dilatih khusus untuk mengenali objek vital bagi navigasi dalam ruangan:
    * Tangga
    * Pintu
    * Orang
    * Meja
    * Kursi
3.  **Estimasi Jarak & Posisi:** Menampilkan jarak objek (dalam meter) dan posisi relatifnya secara akurat.
4.  **Panduan Suara (Text-to-Speech):** Memberikan notifikasi verbal otomatis saat objek terdeteksi (contoh: "Terdapat pintu dalam jarak 2,3 meter", "Awas! Pintu sudah sangat dekat dengan jarak 0.6 meter).
5.  **Umpan Balik Haptik (Getaran):** Memberikan sinyal getaran sebagai peringatan tambahan untuk meningkatkan kewaspadaan pengguna.

## 🧠 Model Kecerdasan Buatan: YOLOv11

Inti dari aplikasi ini ditenagai oleh **YOLOv11 (You Only Look Once v11)**, arsitektur *State-of-the-Art* (SOTA) terbaru dalam deteksi objek.

### Mengapa YOLOv11?
Sesuai dengan analisis kebutuhan proyek, YOLOv11 dipilih karena keunggulannya dibandingkan versi pendahulu (seperti YOLOv8 atau v10):

* **Efisiensi Komputasi Tinggi:** YOLOv11 menawarkan keseimbangan terbaik antara kecepatan (FPS) dan akurasi (mAP). Hal ini krusial karena model harus berjalan lancar di perangkat *mobile* (Android) dengan *resource* terbatas tanpa lag yang membahayakan pengguna.
* **Arsitektur C2f & Modul Attention:** Peningkatan pada blok C2f dan mekanisme *attention* memungkinkan model mendeteksi objek dengan berbagai skala (baik objek besar seperti pintu maupun kecil) dengan lebih presisi.
* **Latency Rendah:** Struktur model yang lebih ringan memungkinkan inferensi yang sangat cepat, memberikan respon *real-time* yang vital untuk aplikasi navigasi keselamatan.

### Implementasi Model
* **Framework:** PyTorch (Training) & TensorFlow Lite (Deployment Android).
* **Dataset:** Dataset kustom yang mencakup total 9.555 citra untuk 5 kelas objek (Tangga, Pintu, Orang, Meja, Kursi) dengan variasi pencahayaan dan sudut pandang.
* **Output:** Bounding box, Class Probability, dan Estimasi Jarak berbasis geometri visual.

## 🛠️ Teknologi yang Digunakan

* **Bahasa Pemrograman:** Kotlin (Android Native).
* **Machine Learning:** Ultralytics YOLOv11.
* **Konversi Model:** TFLite.
* **Tools:** Android Studio, Google Colab (untuk training model).
