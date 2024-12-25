import numpy as np
import matplotlib.pyplot as plt

# ======================
# 1. DATA PERMINTAAN (Qd) & PENAWARAN (Qs)
#    (Contoh FIKTIF, 10 periode)
# ======================
Qd = np.array([90, 85, 88, 91, 94, 100, 103, 98, 105, 102], dtype=float)
Qs = np.array([95, 100, 92, 90, 93,  96,  97, 101,  99, 104], dtype=float)

# Pastikan kedua array punya panjang sama:
N = len(Qd)  # 10

# 2. PERSIAPAN REGRESI
#    Kita akan membentuk model:
#      Qd_{t+1} = alpha*Qd_t + beta*Qs_t
#      Qs_{t+1} = gamma*Qd_t + delta*Qs_t


# (A) Membuat matriks X (input) & y_d (untuk persamaan Qd)
#     Periode t:    t=0..(N-2)  --> memprediksi t+1
#     Artinya kita pakai data Qd[0..N-2], Qs[0..N-2] untuk prediksi Qd[1..N-1]
X = np.column_stack((Qd[:-1], Qs[:-1]))  # (N-1)x2
y_d = Qd[1:]                             # (N-1)x1

# (B) Membuat vektor y_s (untuk persamaan Qs)
y_s = Qs[1:]


# 3. REGRESI LEAST SQUARES

# 3.1. Cari alpha, beta (untuk Qd_{t+1})
theta_d, residuals_d, rank_d, s_d = np.linalg.lstsq(X, y_d, rcond=None)
alpha, beta = theta_d[0], theta_d[1]

# 3.2. Cari gamma, delta (untuk Qs_{t+1})
theta_s, residuals_s, rank_s, s_s = np.linalg.lstsq(X, y_s, rcond=None)
gamma, delta = theta_s[0], theta_s[1]

print("Estimasi parameter:")
print(f"  alpha = {alpha:.4f}")
print(f"  beta  = {beta:.4f}")
print(f"  gamma = {gamma:.4f}")
print(f"  delta = {delta:.4f}")

# 4. BENTUK MATRIKS A & HITUNG NILAI EIGEN

A = np.array([[alpha, beta],
              [gamma, delta]], dtype=float)

eigenvalues, eigenvectors = np.linalg.eig(A)

print("\nMatriks A:")
print(A)
print("\nNilai eigen (lambda):")
for i, lam in enumerate(eigenvalues):
    print(f"  lambda_{i+1} = {lam:.4f}")

# 5. SIMULASI PERIODE KE DEPAN
#    (Misalkan kita lanjutkan 10 periode lagi)

num_future = 10  # jumlah periode tambahan untuk prediksi
t0 = N           # kita mulai prediksi dari data terakhir, t=N (index N-1 di array)

# Array untuk simpan Qd_new dan Qs_new
Qd_sim = np.zeros(N + num_future)
Qs_sim = np.zeros(N + num_future)

# Copy data awal
Qd_sim[:N] = Qd
Qs_sim[:N] = Qs

# Iterasi diskrit:
for t in range(N-1, N + num_future - 1):
    # t berjalan mulai dari 9 -> 9..(9 + num_future-1)
    # Q_{t+1} = A * Q_t
    # Q_t = [Qd_sim[t], Qs_sim[t]]
    Q_next = A.dot(np.array([Qd_sim[t], Qs_sim[t]]))
    Qd_sim[t+1] = Q_next[0]
    Qs_sim[t+1] = Q_next[1]

# 6. PLOTTING

# Buat array sumbu waktu (periode) untuk data asli + simulasi
time_full = np.arange(1, N + num_future + 1)  # 1..(N+num_future)
plt.figure(figsize=(8,5))

plt.plot(time_full, Qd_sim, 'o--', color='blue', label='Permintaan (Qd)')
plt.plot(time_full, Qs_sim, 'o--', color='red',  label='Penawaran (Qs)')

# Tandai garis pemisah antara data aktual dan prediksi
plt.axvline(x=N, color='gray', linestyle=':', label='Mulai Prediksi')

plt.title("Dinamika Permintaan (Qd) & Penawaran (Qs) [Diskrit]")
plt.xlabel("Periode (Minggu ke-)")
plt.ylabel("Kuantitas")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

