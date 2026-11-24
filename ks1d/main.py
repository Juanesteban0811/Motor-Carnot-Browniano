"""
main.py  --  ORQUESTADOR: corre el pipeline completo de KS 1D con ESN paralelo

Flujo:
  1. Simular KS 1D (modulo 1)              -> datos U
  2. Estimar Lambda_max (modulo 2)         -> para medir en tiempos de Lyapunov
  3. Normalizar y partir train/test
  4. Entrenar ESN paralelo (modulo 3)
  5. Evaluar horizonte (modulo 4)
  6. Graficar: mapa espacio-tiempo real vs prediccion + error, y curva RMSE

Ejecutar:  python main.py
Requiere:  numpy, matplotlib, reservoirpy
"""
from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt

from ks_simulacion import simular_ks
from ks_lyapunov import exponente_lyapunov
from esn_paralelo import ESNParalelo
from ks_metricas import evaluar_robusto, horizonte_valido

# ---------------------------------------------------------------- #
#  Parametros del sistema (caso canonico KS 1D, L=22)
# ---------------------------------------------------------------- #
L, Q, DT = 22.0, 64, 0.25        # dominio, puntos de malla, paso de tiempo
N_PASOS = 20000                  # longitud de la serie
UMBRAL = 0.4                     # umbral de RMSE para el horizonte

# ---------------------------------------------------------------- #
#  1) Simular KS 1D
# ---------------------------------------------------------------- #
print("1) Simulando KS 1D...")
U, x = simular_ks(L=L, Q=Q, dt=DT, n_pasos=N_PASOS)
print(f"   datos: {U.shape}  rango [{U.min():.2f}, {U.max():.2f}]")

# ---------------------------------------------------------------- #
#  2) Estimar el exponente de Lyapunov
# ---------------------------------------------------------------- #
print("2) Estimando Lambda_max (puede tardar)...")
lambda_max = exponente_lyapunov(U[0], L=L, Q=Q, dt=DT, n_pasos=10000)
print(f"   Lambda_max = {lambda_max:.4f}  (referencia literatura L=22: ~0.043)")
print(f"   tiempo de Lyapunov = {1.0/lambda_max:.2f}")

# ---------------------------------------------------------------- #
#  3) Normalizar y partir train/test
# ---------------------------------------------------------------- #
n_train = int(0.75 * N_PASOS)
U_train, U_test = U[:n_train], U[n_train:]
media, std = U_train.mean(axis=0), U_train.std(axis=0)
U_train_norm = (U_train - media) / std
U_test_norm = (U_test - media) / std

# ---------------------------------------------------------------- #
#  4) Entrenar ESN paralelo
# ---------------------------------------------------------------- #
print("4) Entrenando ESN paralelo...")
esn = ESNParalelo(Q=Q, n_grupos=16, solape=4, units_g=600,
                  lr=1.0, sr=0.4, input_scaling=0.1, ridge=1e-5,
                  noise_amp=0.005, seed=42, clip=6.0)
esn.entrenar(U_train_norm, warmup=100)

# ---------------------------------------------------------------- #
#  5) Evaluar horizonte (robusto, varias ventanas)
# ---------------------------------------------------------------- #
print("5) Evaluando horizonte...")
h_med, h_std, h_lista = evaluar_robusto(esn, U_test_norm, DT, lambda_max, umbral=UMBRAL)
print(f"   Horizonte: {h_med:.2f} +/- {h_std:.2f} tiempos de Lyapunov "
      f"(sobre {len(h_lista)} ventanas)")

# ---------------------------------------------------------------- #
#  6) Graficar una ventana representativa
# ---------------------------------------------------------------- #
step_warmup, step_prediccion = 150, 300
inicio = step_warmup
pred_norm = esn.predecir(U_test_norm, inicio, step_warmup, step_prediccion)
real_norm = U_test_norm[inicio:inicio + step_prediccion]
h_ventana, cruce, err = horizonte_valido(pred_norm, real_norm, DT, lambda_max, UMBRAL)

Y_pred = pred_norm * std + media
Y_real = real_norm * std + media

t_lyap = 1.0 / lambda_max
tiempo_lyap = np.arange(step_prediccion) * DT / t_lyap     # eje en tiempos de Lyapunov

fig, axes = plt.subplots(3, 1, figsize=(11, 10))

# --- mapa espacio-tiempo: real ---
im0 = axes[0].imshow(Y_real.T, aspect="auto", origin="lower", cmap="jet",
                     extent=[0, tiempo_lyap[-1], 0, Q])
axes[0].set_title("KS 1D real")
axes[0].set_ylabel("espacio (x)")
fig.colorbar(im0, ax=axes[0])

# --- mapa espacio-tiempo: prediccion ---
im1 = axes[1].imshow(Y_pred.T, aspect="auto", origin="lower", cmap="jet",
                     extent=[0, tiempo_lyap[-1], 0, Q])
axes[1].set_title("Prediccion ESN paralelo")
axes[1].set_ylabel("espacio (x)")
fig.colorbar(im1, ax=axes[1])

# --- curva de error con eje en tiempos de Lyapunov ---
axes[2].plot(tiempo_lyap, err, color="blue")
axes[2].axhline(UMBRAL, color="gray", ls="--", label=f"umbral={UMBRAL}")
axes[2].axvline(h_ventana, color="red", ls=":",
                label=f"valido: {h_ventana:.1f} tiempos de Lyapunov")
axes[2].set_xlabel("tiempo (en tiempos de Lyapunov, $\\Lambda t$)")
axes[2].set_ylabel("RMSE normalizado")
axes[2].set_title("Error vs tiempo de Lyapunov")
axes[2].legend()

plt.tight_layout()
plt.savefig("prediccion_ks1d.png", dpi=120)
plt.show()
print("Figura guardada en prediccion_ks1d.png")
