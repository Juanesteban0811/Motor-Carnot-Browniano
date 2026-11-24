# KS 1D + ESN Paralelo

Predicción de la ecuación de **Kuramoto-Sivashinsky 1D** con **Echo State Networks paralelos**,
reproduciendo el método de Pathak et al. (PRL 120, 024102, 2018).

Esta es la versión **1D** (la del artículo original), que reemplaza el intento previo en 2D:
1D es el sistema que el paper resuelve, es más fácil de simular y el ESN paralelo rinde mucho mejor.

## Estructura (bien segmentada)

| Archivo | Rol |
|---------|-----|
| `ks_simulacion.py` | **Módulo 1** — simula KS 1D con método espectral ETDRK4 (solo numpy, sin shenfun) |
| `ks_lyapunov.py`   | **Módulo 2** — estima el exponente de Lyapunov máximo (dos trayectorias) |
| `esn_paralelo.py`  | **Módulo 3** — ESN paralelo: reservoirs locales, readout `[r;r²]`, solape, clip |
| `ks_metricas.py`   | **Módulo 4** — RMSE por paso y horizonte en tiempos de Lyapunov |
| `main.py`          | **Orquestador** — corre todo el pipeline y grafica |

## Cómo correrlo

```bash
pip install numpy matplotlib reservoirpy
cd ks1d
python main.py
```

Genera `prediccion_ks1d.png` con el mapa espacio-tiempo real vs. predicción y la curva de error
en tiempos de Lyapunov.

## Parámetros del caso canónico

- Dominio `L = 22`, `Q = 64` puntos, `dt = 0.25` → sistema caótico estándar.
- Exponente de Lyapunov de referencia (literatura, L=22): **Λ_max ≈ 0.043**.
  El módulo 2 lo estima; debería salir cercano a ese valor.
- ESN paralelo: 16 grupos de 4 variables, solape 4, 600 neuronas por grupo.

## Lo que se aprovecha de todo lo aprendido

- **Readout `[r ; r²]`** (cuadrados puros del estado, sin cruzados) — fiel al artículo.
- **Reservoirs locales paralelos** con solape periódico — la contribución clave del paper.
- **Clip de seguridad** en la predicción autónoma — evita las explosiones numéricas que vimos.
- **Métrica en tiempos de Lyapunov** con promedio robusto sobre varias ventanas.

## Notas

- El `solape` y el `n_grupos` se pueden ajustar en `main.py` (el óptimo se encuentra barriendo,
  como se hizo en Lorenz-96: existe un solape que corresponde al alcance del acoplamiento).
- El exponente de Lyapunov aquí **sí se estima del propio sistema** (módulo 2), así que el
  horizonte en tiempos de Lyapunov es correcto (a diferencia del intento 2D, donde era un placeholder).
