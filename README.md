# Porous Media Flow PINN GPU

Simulasi aliran air pada media berpori dua dimensi menggunakan dua pendekatan yang dijalankan dalam satu notebook:

- Finite Difference Method (FDM) sebagai solusi referensi numerik.
- Physics-Informed Neural Network (PINN) berbasis PyTorch sebagai pendekatan pembelajaran mesin yang mematuhi persamaan fisika.

Proyek ini difokuskan pada studi aliran air di tanah rawa atau akuifer dangkal dengan forcing infiltrasi hujan multi-titik, domain 2D, serta pelatihan PINN di GPU jika CUDA tersedia.

## Ringkasan

Notebook utama proyek ini adalah [`porous_media_flow_pinn_gpu.ipynb`](./porous_media_flow_pinn_gpu.ipynb). Di dalamnya, seluruh pipeline simulasi dibangun dari awal:

1. konfigurasi environment, reproducibility, dan deteksi GPU,
2. definisi domain fisik dan parameter hidrologi,
3. pembangkitan pola hujan spasial-temporal multi-titik,
4. penyelesaian PDE menggunakan FDM explicit 2D,
5. pembangunan model PINN dengan PyTorch,
6. pembuatan data observasi dummy berbasis hasil FDM,
7. training PINN dengan kombinasi loss fisika dan data,
8. evaluasi error PINN terhadap solusi referensi FDM,
9. visualisasi snapshot dan pembuatan animasi GIF.

Secara praktis, notebook ini dapat dipakai sebagai:

- bahan pembelajaran porous media flow,
- demonstrasi integrasi solver numerik klasik dan PINN,
- template awal eksperimen groundwater / soil-water flow berbasis data,
- dasar pengembangan inverse PINN untuk estimasi parameter dari observasi lapangan.

## Permasalahan yang Dimodelkan

Model menyelesaikan persamaan difusi 2D sederhana untuk hydraulic head `h(x, y, t)`:

```math
S_s \frac{\partial h}{\partial t} - K \left(\frac{\partial^2 h}{\partial x^2} + \frac{\partial^2 h}{\partial y^2}\right) - R(x,y,t) = 0
```

dengan:

- `h` = hydraulic head,
- `S_s` = specific storage,
- `K` = hydraulic conductivity,
- `R(x,y,t)` = source term infiltrasi hujan.

Dalam implementasi FDM di notebook, bentuk yang dipakai ekuivalen dengan:

```math
\frac{\partial h}{\partial t} = D \left(\frac{\partial^2 h}{\partial x^2} + \frac{\partial^2 h}{\partial y^2}\right) + \frac{R}{S_s}
```

dengan `D = K / S_s`.

## Parameter Simulasi Saat Ini

Parameter default yang tertanam di notebook:

| Parameter | Nilai | Keterangan |
| --- | --- | --- |
| Domain | `50 m x 50 m` | Area simulasi 2D |
| Durasi simulasi | `6 jam` | Kejadian hujan jangka pendek |
| Grid spasial | `51 x 51` | Resolusi `1 m` per arah |
| Head awal | `1.0 m` | Kondisi awal seragam |
| Head batas | `1.0 m` | Dirichlet boundary condition di semua sisi |
| `K_day` | `0.8 m/hari` | Konduktivitas hidrolik awal |
| `K` | `0.03333 m/jam` | Hasil konversi dari `K_day` |
| `S_s` | `0.15` | Specific storage |
| Resolusi waktu target | `0.10 jam` | Disesuaikan lagi oleh syarat stabilitas |
| Titik hujan | `18 titik acak` | Forcing infiltrasi multi-sumber |

Notebook juga menghitung `dt` berdasarkan syarat stabilitas skema explicit 2D:

```text
rx + ry <= 0.5
```

sehingga simulasi FDM tetap berada pada rentang yang aman.

## Isi Notebook Secara Detail

### 1. Import library dan konfigurasi GPU

Notebook memuat library utama berikut:

- `numpy`
- `matplotlib`
- `torch`
- `psutil`
- `imageio`

Pada tahap ini juga dilakukan:

- pengaturan seed untuk hasil yang reproducible,
- aktivasi `torch.set_float32_matmul_precision("high")`,
- pemilihan device `cuda` atau `cpu`,
- penampilan informasi RAM dan VRAM untuk membantu pemantauan resource.

### 2. Definisi domain fisik dan parameter hidrologi

Bagian ini menetapkan:

- ukuran domain `Lx`, `Ly`,
- jumlah grid `Nx`, `Ny`,
- durasi simulasi `T_end`,
- parameter `K`, `S_s`, dan `D`,
- grid koordinat `X`, `Y`,
- array waktu `t_arr`.

Karena notebook memakai solver FDM explicit, nilai `dt` dihitung dari batas stabilitas numerik, bukan hanya dari target resolusi waktu.

### 3. Forcing hujan multi-titik

Salah satu komponen penting proyek ini adalah source term infiltrasi hujan yang tidak seragam. Hujan dimodelkan sebagai gabungan beberapa sel hujan acak dengan:

- pusat hujan acak di domain,
- lebar sebaran spasial (`sigma`) yang berbeda,
- intensitas tiap sel hujan yang berbeda,
- profil temporal piecewise-linear yang berubah terhadap waktu.

Implementasi tersedia dalam dua versi:

- `rainfall_source_np(...)` untuk solver FDM,
- `rainfall_source_torch(...)` untuk residual PINN.

Dengan desain ini, forcing hujan bersifat:

- heterogen secara spasial,
- berubah terhadap waktu,
- konsisten antara solver numerik dan model PINN.

### 4. Kondisi awal dan kondisi batas

Kondisi awal:

- hydraulic head seragam `1.0 m` di seluruh domain.

Kondisi batas:

- Dirichlet boundary condition konstan `1.0 m` di keempat sisi domain.

Model saat ini sengaja dibuat sederhana agar fokus eksperimen ada pada pengaruh source term hujan dan pembandingan FDM vs PINN.

### 5. Solver FDM

FDM dipakai sebagai solusi referensi. Notebook:

- membangun tensor solusi `h[t, x, y]`,
- menghitung indikator stabilitas `rx` dan `ry`,
- melakukan update explicit pada node interior,
- menerapkan boundary Dirichlet di tiap time step.

Kelebihan pendekatan ini di proyek:

- mudah diverifikasi,
- langsung selaras dengan PDE yang sama,
- dapat dipakai untuk menghasilkan data pseudo-observasi bagi PINN.

### 6. Arsitektur PINN

Model PINN dibangun menggunakan `torch.nn.Module` dengan karakteristik:

- input `3` dimensi: `x`, `y`, `t`,
- hidden size `96`,
- `5` hidden layer,
- aktivasi `Tanh`,
- output `1` dimensi: prediksi `h`.

Input dinormalisasi terhadap `Lx`, `Ly`, dan `T_end` agar training lebih stabil.

Residual PDE dihitung menggunakan autograd PyTorch untuk memperoleh:

- turunan pertama terhadap `x`, `y`, `t`,
- turunan kedua terhadap `x` dan `y`,
- residual fisika yang menjadi komponen utama loss.

### 7. Sampling data training

Training PINN memakai tiga kelompok titik:

- titik interior domain untuk loss PDE,
- titik waktu awal untuk initial condition,
- titik di seluruh sisi domain untuk boundary condition.

Selain itu, notebook membangkitkan data observasi dummy dari hasil FDM:

- `12` sensor spasial,
- observasi pada jam ke-`1`, `2`, dan `3`,
- bias kecil per jam,
- noise Gaussian kecil.

Data dummy ini berguna untuk mendemonstrasikan skenario PINN yang tidak hanya berbasis persamaan fisika, tetapi juga mengonsumsi observasi.

### 8. Training PINN

Training dilakukan dalam dua tahap:

1. `Adam` untuk optimasi awal,
2. `LBFGS` untuk refinement akhir.

Konfigurasi default yang dipakai notebook:

| Komponen | Nilai |
| --- | --- |
| `epochs_adam` | `8000` |
| `epochs_lbfgs` | `300` |
| `n_int` | `2500` |
| `n_ini` | `1200` |
| `n_bnd` | `1200` |
| `lr` | `8e-4` |
| `w_pde` | `1.0` |
| `w_ini` | `25.0` |
| `w_bnd` | `25.0` |
| `w_data` | `30.0` |

Loss total merupakan kombinasi dari:

- loss residual PDE,
- loss initial condition,
- loss boundary condition,
- loss data observasi.

Notebook juga menyimpan `history` loss untuk divisualisasikan setelah training.

### 9. Evaluasi hasil

Setelah training, model PINN dievaluasi pada grid penuh untuk waktu tertentu, lalu dibandingkan dengan solusi FDM. Notebook menampilkan:

- kontur head hasil PINN,
- kontur head referensi FDM,
- peta galat absolut `|PINN - FDM|`,
- metrik RMSE,
- rata-rata error absolut.

Bagian ini penting untuk menilai seberapa baik PINN merekonstruksi solusi numerik referensi.

### 10. Visualisasi dan animasi

Notebook menyediakan utilitas visualisasi untuk:

- plot field 2D,
- plot loss history,
- overlay titik hujan,
- overlay titik sensor dummy.

Selain itu, notebook membuat:

- frame PNG untuk hasil FDM,
- frame PNG untuk hasil PINN,
- `fdm_animation.gif`,
- `pinn_animation.gif`.

Jika dijalankan penuh, notebook juga akan membuat direktori output:

- `frames_fdm/`
- `frames_pinn/`

## Struktur Repository

Saat ini isi repository sangat ringkas:

```text
.
|-- porous_media_flow_pinn_gpu.ipynb
|-- README.md
`-- LICENSE
```

Direktori tambahan seperti `frames_fdm/` dan `frames_pinn/` akan muncul setelah sel animasi dijalankan.

## Cara Menjalankan

### 1. Siapkan environment Python

Disarankan Python `3.10+`.

Install dependensi:

```bash
pip install numpy matplotlib torch psutil imageio notebook jupyter
```

Jika ingin memakai GPU, install PyTorch yang sesuai dengan versi CUDA di sistem Anda dari dokumentasi resmi PyTorch.

### 2. Jalankan Jupyter Notebook

```bash
jupyter notebook
```

lalu buka file:

```text
porous_media_flow_pinn_gpu.ipynb
```

### 3. Eksekusi sel secara berurutan

Urutan yang disarankan:

1. import dan konfigurasi device,
2. parameter domain,
3. hujan, kondisi awal, dan batas,
4. solver FDM,
5. model PINN,
6. sampling data,
7. training,
8. evaluasi,
9. animasi.

Menjalankan sel secara berurutan penting karena banyak variabel global notebook saling bergantung.

## Output yang Dihasilkan

Jika notebook dijalankan penuh, Anda akan memperoleh:

- informasi device, RAM, dan VRAM,
- snapshot distribusi infiltrasi hujan,
- snapshot hydraulic head hasil FDM,
- kurva training loss PINN,
- snapshot prediksi PINN,
- peta error terhadap FDM,
- GIF animasi evolusi FDM,
- GIF animasi evolusi PINN.

## Kelebihan Proyek Ini

- Menggabungkan metode numerik klasik dan PINN dalam satu workflow.
- Forcing hujan dibuat cukup realistis karena multi-titik dan berubah terhadap waktu.
- Sudah siap memanfaatkan GPU untuk training.
- Menyediakan pseudo-observasi untuk simulasi data-driven.
- Visualisasi cukup lengkap untuk analisis kualitatif dan kuantitatif.

## Batasan Saat Ini

- Repository masih berupa satu notebook monolitik, belum dipisah menjadi modul Python.
- Data observasi yang dipakai untuk training masih dummy, bukan data lapangan nyata.
- Parameter tanah masih homogen dan isotropik.
- Boundary condition masih seragam di semua sisi.
- Model fisik masih difusi sederhana, belum memasukkan proses hidrologi yang lebih kompleks.

## Arah Pengembangan

Pengembangan lanjutan yang sudah tersirat dari notebook:

- permeabilitas heterogen `K = K(x, y)`,
- anisotropi `Kx != Ky`,
- boundary sungai atau kanal,
- evapotranspirasi,
- data curah hujan riil,
- penggantian observasi dummy dengan data piezometer/sumur pantau,
- inverse PINN untuk estimasi parameter dari data observasi.

## Saran Pengembangan Repository

Jika proyek ini ingin dikembangkan lebih jauh, struktur berikut akan lebih mudah dipelihara:

```text
.
|-- notebooks/
|-- src/
|   |-- fdm_solver.py
|   |-- pinn_model.py
|   |-- rainfall.py
|   `-- visualization.py
|-- outputs/
|-- README.md
`-- LICENSE
```

Dengan pemisahan tersebut, eksperimen akan lebih mudah diuji, didokumentasikan, dan direproduksi.

## Lisensi

Proyek ini menggunakan lisensi [MIT](./LICENSE).
