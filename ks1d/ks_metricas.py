"""
ks_metricas.py  --  MODULO 4: metricas de evaluacion

Mide el horizonte de prediccion en TIEMPOS DE LYAPUNOV (como el articulo):
  - RMSE normalizado por paso
  - "tiempo de prediccion valido" = primer instante en que el RMSE cruza un umbral
  - promedio robusto sobre varias ventanas de arranque
"""
from __future__ import annotations

import numpy as np


def rmse_por_paso(pred_norm, real_norm):
    """RMSE normalizado en cada paso de tiempo (promedio sobre las variables)."""
    return np.sqrt(((pred_norm - real_norm) ** 2).mean(axis=1))


def horizonte_valido(pred_norm, real_norm, dt_datos, lambda_max, umbral=0.4):
    """
    Devuelve el horizonte de prediccion valido en tiempos de Lyapunov.

    dt_datos   : paso de tiempo entre filas de los datos
    lambda_max : exponente de Lyapunov maximo del sistema
    umbral     : umbral de RMSE normalizado (estandar 0.4)
    """
    err = rmse_por_paso(pred_norm, real_norm)
    cruce = np.argmax(err > umbral) if np.any(err > umbral) else len(err)
    t_lyap = 1.0 / lambda_max
    return (cruce * dt_datos) / t_lyap, cruce, err


def evaluar_robusto(esn, U_test_norm, dt_datos, lambda_max,
                    step_warmup=150, step_prediccion=300, umbral=0.4, n_ventanas=6):
    """
    Promedia el horizonte sobre varias ventanas de arranque (metrica robusta).
    Retorna (media, desviacion, lista_de_horizontes).
    """
    arranques = [step_warmup + k * 300 for k in range(n_ventanas)
                 if step_warmup + k * 300 + step_prediccion <= len(U_test_norm)]
    horizontes = []
    for inicio in arranques:
        pred = esn.predecir(U_test_norm, inicio, step_warmup, step_prediccion)
        real = U_test_norm[inicio:inicio + step_prediccion]
        h, _, _ = horizonte_valido(pred, real, dt_datos, lambda_max, umbral)
        horizontes.append(h)
    horizontes = np.array(horizontes)
    return horizontes.mean(), horizontes.std(), horizontes
