"""
ks_simulacion.py  --  MODULO 1: generacion de datos de Kuramoto-Sivashinsky 1D

Simula la ecuacion de Kuramoto-Sivashinsky en 1D (la misma que usa Pathak et al.,
PRL 120, 024102, 2018):

    y_t = -y*y_x - y_xx - y_xxxx       (dominio periodico de longitud L)

Se integra con un metodo espectral ETDRK4 (Kassam & Trefethen, 2005), que es el
estandar para esta ecuacion porque es "stiff" (los terminos y_xx e y_xxxx la hacen
rigida). NO se usa shenfun: todo con numpy (FFT), mas simple y portable.

Caso canonico reproducible: L = 22, Q = 64 puntos -> sistema caotico con
exponente de Lyapunov maximo de referencia en la literatura: Lambda ~ 0.043.

Salida: matriz U de forma (n_tiempos, Q), filas = tiempo, columnas = puntos del espacio.
"""
from __future__ import annotations

import numpy as np


def simular_ks(L=22.0, Q=64, dt=0.25, n_pasos=20000, transitorio=2000, semilla=1):
    """
    Integra KS 1D con ETDRK4 y devuelve la serie temporal del campo.

    Parametros
    ----------
    L           : longitud del dominio periodico (L=22 -> caotico canonico)
    Q           : numero de puntos de la malla espacial (potencia de 2 para la FFT)
    dt          : paso de tiempo
    n_pasos     : numero de pasos que se guardan (filas de la salida)
    transitorio : pasos iniciales que se descartan (spin-up, para caer en el atractor)
    semilla     : semilla del ruido de la condicion inicial

    Retorna
    -------
    U : np.ndarray (n_pasos, Q)  campo y(x,t); filas = tiempo, columnas = espacio
    x : np.ndarray (Q,)          coordenadas espaciales
    """
    rng = np.random.default_rng(semilla)

    # --- malla espacial y numeros de onda ---
    x = L * np.arange(Q) / Q
    k = 2.0 * np.pi * np.fft.fftfreq(Q, d=L / Q)   # numeros de onda

    # --- operador lineal en espacio de Fourier: L_hat = k^2 - k^4 ---
    #     (viene de -y_xx - y_xxxx  ->  +k^2 - k^4 en Fourier)
    Lop = k**2 - k**4

    # --- coeficientes ETDRK4 (Kassam & Trefethen 2005) via integral de contorno ---
    E = np.exp(dt * Lop)
    E2 = np.exp(dt * Lop / 2.0)
    M = 32                                   # puntos del contorno para la cuadratura
    r = np.exp(1j * np.pi * (np.arange(1, M + 1) - 0.5) / M)
    LR = dt * Lop[:, None] + r[None, :]
    Q_c = dt * np.real(np.mean((np.exp(LR / 2.0) - 1.0) / LR, axis=1))
    f1 = dt * np.real(np.mean((-4.0 - LR + np.exp(LR) * (4.0 - 3.0 * LR + LR**2)) / LR**3, axis=1))
    f2 = dt * np.real(np.mean((2.0 + LR + np.exp(LR) * (-2.0 + LR)) / LR**3, axis=1))
    f3 = dt * np.real(np.mean((-4.0 - 3.0 * LR - LR**2 + np.exp(LR) * (4.0 - LR)) / LR**3, axis=1))

    # --- condicion inicial: pequena perturbacion aleatoria ---
    y0 = 0.01 * rng.standard_normal(Q)
    v = np.fft.fft(y0)                        # estado en espacio de Fourier

    # termino no lineal: N(y) = -0.5 * (y^2)_x  ->  en Fourier: -0.5 * i*k * fft(y^2)
    g = -0.5j * k

    def paso_etdrk4(v):
        Nv = g * np.fft.fft(np.real(np.fft.ifft(v)) ** 2)
        a = E2 * v + Q_c * Nv
        Na = g * np.fft.fft(np.real(np.fft.ifft(a)) ** 2)
        b = E2 * v + Q_c * Na
        Nb = g * np.fft.fft(np.real(np.fft.ifft(b)) ** 2)
        c = E2 * a + Q_c * (2.0 * Nb - Nv)
        Nc = g * np.fft.fft(np.real(np.fft.ifft(c)) ** 2)
        return E * v + Nv * f1 + 2.0 * (Na + Nb) * f2 + Nc * f3

    # --- descartar transitorio ---
    for _ in range(transitorio):
        v = paso_etdrk4(v)

    # --- integrar y guardar ---
    U = np.zeros((n_pasos, Q))
    for i in range(n_pasos):
        v = paso_etdrk4(v)
        U[i] = np.real(np.fft.ifft(v))

    return U, x


if __name__ == "__main__":
    U, x = simular_ks()
    print("Forma de los datos (tiempo, espacio):", U.shape)
    print("Rango de valores: [%.2f, %.2f]  media: %.3f" % (U.min(), U.max(), U.mean()))
    np.save("ks1d_datos.npy", U)
    print("Guardado en ks1d_datos.npy")
