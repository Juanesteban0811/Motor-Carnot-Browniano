"""
ks_lyapunov.py  --  MODULO 2: exponente de Lyapunov maximo de KS 1D

Estima Lambda_max con el metodo de las dos trayectorias (el mismo que se uso para
Lorenz-96): se integra la trayectoria y una copia perturbada, se mide la separacion
cada cierto numero de pasos, se renormaliza, y se promedia el crecimiento logaritmico.

    Lambda = (1 / tiempo_total) * sum( ln( d / d0 ) )

Para KS 1D con L=22 el valor de referencia en la literatura (Pathak et al. 2018;
Vlachas et al.) es Lambda_max ~ 0.043. Sirve para VERIFICAR esta estimacion.

Reusa el integrador ETDRK4 del modulo de simulacion para que la dinamica sea identica.
"""
from __future__ import annotations

import numpy as np


def _construir_etdrk4(L, Q, dt):
    """Devuelve una funcion paso_etdrk4(v) y utilidades, identica a la simulacion."""
    k = 2.0 * np.pi * np.fft.fftfreq(Q, d=L / Q)
    Lop = k**2 - k**4
    E = np.exp(dt * Lop)
    E2 = np.exp(dt * Lop / 2.0)
    M = 32
    r = np.exp(1j * np.pi * (np.arange(1, M + 1) - 0.5) / M)
    LR = dt * Lop[:, None] + r[None, :]
    Q_c = dt * np.real(np.mean((np.exp(LR / 2.0) - 1.0) / LR, axis=1))
    f1 = dt * np.real(np.mean((-4.0 - LR + np.exp(LR) * (4.0 - 3.0 * LR + LR**2)) / LR**3, axis=1))
    f2 = dt * np.real(np.mean((2.0 + LR + np.exp(LR) * (-2.0 + LR)) / LR**3, axis=1))
    f3 = dt * np.real(np.mean((-4.0 - 3.0 * LR - LR**2 + np.exp(LR) * (4.0 - LR)) / LR**3, axis=1))
    g = -0.5j * k

    def paso(v):
        Nv = g * np.fft.fft(np.real(np.fft.ifft(v)) ** 2)
        a = E2 * v + Q_c * Nv
        Na = g * np.fft.fft(np.real(np.fft.ifft(a)) ** 2)
        b = E2 * v + Q_c * Na
        Nb = g * np.fft.fft(np.real(np.fft.ifft(b)) ** 2)
        c = E2 * a + Q_c * (2.0 * Nb - Nv)
        Nc = g * np.fft.fft(np.real(np.fft.ifft(c)) ** 2)
        return E * v + Nv * f1 + 2.0 * (Na + Nb) * f2 + Nc * f3

    return paso


def exponente_lyapunov(y0, L=22.0, Q=64, dt=0.25, n_pasos=20000,
                       d0=1e-7, renorm=5):
    """
    Estima Lambda_max por dos trayectorias.

    y0      : condicion inicial en espacio REAL (np.ndarray de tamano Q)
    d0      : separacion inicial entre trayectorias
    renorm  : cada cuantos pasos se mide la separacion y se renormaliza

    Retorna Lambda_max (float).
    """
    paso = _construir_etdrk4(L, Q, dt)

    v = np.fft.fft(y0)
    pert = np.zeros(Q)
    pert[0] = d0                              # perturbacion minuscula
    w = np.fft.fft(y0 + pert)

    suma = 0.0
    n_renorm = 0
    for i in range(1, n_pasos + 1):
        v = paso(v)
        w = paso(w)
        if i % renorm == 0:
            yv = np.real(np.fft.ifft(v))
            yw = np.real(np.fft.ifft(w))
            d = np.linalg.norm(yw - yv)
            if d > 0:
                suma += np.log(d / d0)
                n_renorm += 1
                # devolver w a distancia d0 de v (renormalizacion)
                yw = yv + (yw - yv) * (d0 / d)
                w = np.fft.fft(yw)

    return suma / (n_renorm * renorm * dt)


if __name__ == "__main__":
    from ks_simulacion import simular_ks
    U, x = simular_ks(n_pasos=1000)           # un tramo corto solo para la condicion inicial
    Lambda = exponente_lyapunov(U[-1])
    print(f"Lambda_max estimado: {Lambda:.4f}")
    print(f"Tiempo de Lyapunov 1/Lambda: {1.0/Lambda:.2f}")
    print("Referencia literatura (L=22): ~0.043")
