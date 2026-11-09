"""
termodinamica.py  --  SEGMENTO 4  (Termodinamica estocastica)
Dueño: Integrante 4

Lee el CSV de trayectorias generado por el nucleo C++ (Segmento 3) y calcula
las cantidades termodinamicas del motor de Carnot browniano, con las formulas
de Martinez et al. (Nature Physics 2015, Metodos):

  - Trabajo (integral de Stratonovich):
        dW = 1/2 [ F_k(x_t) + F_k(x_{t+dt}) ] * (kappa_{t+dt} - kappa_t),
        con F_k = x^2 / 2.
  - Calor (Sekimoto, 1a ley):
        dQ = dU + dE_cin - dW,
        dU = 1/2 (kappa_{t+dt} x_{t+dt}^2 - kappa_t x_t^2),
        dE_cin = (m/2) (v_{t+dt}^2 - v_t^2).
  - Eficiencia (ec. 1):
        eta = -W / (Q2 + Q3 + Q4)
  - Potencia:
        P = -<W> / tau

TODO (Integrante 4):
  Sesion 1: implementar leer_csv() (ya con datos de prueba si C++ no esta).
  Sesion 2: implementar trabajo_por_tramo() con Stratonovich.
  Sesion 3: implementar calor_por_tramo() (necesita v -> dE_cin).
  Sesion 4: implementar eficiencia() / potencia() y verificar eta_C y eta_CA.
  Sesion 5: exportar la tabla-resumen por cada tau.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

# Columnas esperadas del CSV (contrato con C++, NO cambiar sin avisar al grupo).
COLUMNAS = ["realizacion", "tiempo", "tramo", "x", "v", "T", "kappa"]


@dataclass
class ResultadoTermo:
    """Resumen termodinamico de una corrida (un valor de tau)."""
    tau: float
    W_medio: float = float("nan")      # <W_tau>
    Q_medio: float = float("nan")      # <Q_tau>
    Q_por_tramo: tuple = ()            # (<Q1>, <Q2>, <Q3>, <Q4>)
    eficiencia: float = float("nan")   # eta
    potencia: float = float("nan")     # P = -<W>/tau


def leer_csv(ruta: str) -> pd.DataFrame:
    """Carga el CSV de trayectorias. STUB: validar columnas y devolver df."""
    # TODO (Sesion 1): leer con pandas y validar COLUMNAS.
    df = pd.read_csv(ruta)
    faltan = [c for c in COLUMNAS if c not in df.columns]
    if faltan:
        raise ValueError(f"Al CSV le faltan columnas: {faltan}")
    return df


def _fuerza_conjugada(x: np.ndarray) -> np.ndarray:
    """F_kappa = x^2 / 2  (fuerza conjugada al parametro de control)."""
    return 0.5 * x**2


def trabajo_por_tramo(df_realizacion: pd.DataFrame) -> dict[int, float]:
    """Trabajo (Stratonovich) acumulado por tramo en UNA realizacion.

    STUB (Sesion 1): devuelve ceros. Implementar en Sesion 2.
    """
    # TODO (Sesion 2):
    #   x, kappa = df["x"].values, df["kappa"].values
    #   Fk = _fuerza_conjugada(x)
    #   dW = 0.5 * (Fk[:-1] + Fk[1:]) * np.diff(kappa)
    #   sumar dW agrupando por df["tramo"].
    return {1: 0.0, 2: 0.0, 3: 0.0, 4: 0.0}


def calor_por_tramo(df_realizacion: pd.DataFrame, masa: float) -> dict[int, float]:
    """Calor (Sekimoto) por tramo en UNA realizacion: dQ = dU + dE_cin - dW.

    STUB (Sesion 1): devuelve ceros. Implementar en Sesion 3.
    """
    # TODO (Sesion 3):
    #   dU     = 0.5*(kappa[1:]*x[1:]**2 - kappa[:-1]*x[:-1]**2)
    #   dEcin  = 0.5*masa*(v[1:]**2 - v[:-1]**2)
    #   dQ     = dU + dEcin - dW
    return {1: 0.0, 2: 0.0, 3: 0.0, 4: 0.0}


def eficiencia(W_total: float, Q_por_tramo: dict[int, float]) -> float:
    """eta = -W / (Q2 + Q3 + Q4)   (ec. 1 del paper).

    STUB (Sesion 1). Implementar en Sesion 4 (incluir variantes eta1, eta2).
    """
    # TODO (Sesion 4)
    return float("nan")


def potencia(W_medio: float, tau: float) -> float:
    """P = -<W> / tau."""
    # TODO (Sesion 4)
    return -W_medio / tau if tau else float("nan")


def analizar(ruta_csv: str, tau: float, masa: float) -> ResultadoTermo:
    """Pipeline completo para un CSV: promedia sobre realizaciones.

    STUB (Sesion 1): arma el objeto resultado con valores vacios para que el
    Segmento 5 pueda graficar con datos de prueba. Completar en Sesiones 2-5.
    """
    df = leer_csv(ruta_csv)
    # TODO (Sesiones 2-5): agrupar por 'realizacion', calcular W y Q por tramo,
    # promediar, y rellenar ResultadoTermo. Verificar en Sesion 4:
    #   eta -> eta_C = 1 - T_c/T_h ~ 0.43 en el limite cuasiestatico
    #   eta* ~ eta_CA = 1 - sqrt(T_c/T_h) ~ 0.25 a potencia maxima
    return ResultadoTermo(tau=tau)


if __name__ == "__main__":
    # Prueba manual rapida (requiere datos/trayectorias.csv del nucleo C++).
    res = analizar("datos/trayectorias.csv", tau=200.0, masa=1.0e-9)
    print(res)
