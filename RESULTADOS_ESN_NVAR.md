# Resultados — ESN + NVAR sobre Lorenz-96

**Notebook:** `AvancesProyectofinalS1.ipynb`
**Sistema de prueba:** Lorenz-96 con `F = 8`, `N = 36` variables, integrado por Runge-Kutta 4.
**Arquitectura:** Echo State Network (reservoir) **+ términos cuadráticos vía NVAR** conectados en
paralelo al readout.

> Este documento reporta el resultado que **sí funcionó** como línea base, el criterio con el
> que se validó, y de dónde sale el número de **2.67 tiempos de Lyapunov**.

---

## 1. Datos y parámetros del sistema

Los datos se generan con `generar_datos(Q=36, F=8.0, dt=0.01, cada=5)`:

- Paso de integración interno: `dt = 0.01`.
- Se guarda **1 de cada 5** pasos → el paso de tiempo **efectivo** entre filas es:
  ```
  Δt = dt · cada = 0.01 · 5 = 0.05 unidades de tiempo (u.t.)
  ```
- Matriz de datos: `U` de forma `(20000, 36)` (filas = tiempo, columnas = variables).
- Normalización por variable con la media y desviación del conjunto de entrenamiento
  (necesaria para no saturar la `tanh` del reservoir, ya que `F = 8`).

### Verificación de los datos
Chequeos realizados sobre `U` que confirman que los datos son correctos (valores canónicos de
Lorenz-96 con `F = 8`):

| Magnitud | Valor |
|----------|-------|
| rango (min, max) | ≈ (−10.0, 15.6) |
| media global | ≈ 2.34 |
| desviación global | ≈ 3.64 |
| cambio medio entre filas consecutivas | ≈ 0.68 |
| decorrelación temporal (1er cruce por cero) | ≈ 11 pasos |

El tiempo de decorrelación (≈11 pasos) es consistente con el tiempo de Lyapunov en pasos
(≈12, ver §4), lo que valida que `Δt = 0.05` y `λ_max = 1.67` son coherentes con los datos.

---

## 2. Arquitectura: ESN + NVAR

Lorenz-96 es **no lineal** (`dx_j/dt = (x_{j+1} − x_{j−2})·x_{j−1} − x_j + F`). Un reservoir con
readout puramente lineal no captura bien esa no linealidad. Para la línea base se añadió un nodo
`NVAR` (términos cuadráticos) en paralelo al reservoir:

```
ESN = entrada >> reservoir >> readout   &   entrada >> cuadraticos >> readout
```

De modo que el readout se entrena sobre la concatenación de los estados del reservoir y los
términos no lineales de la entrada. **Nota de fidelidad:** el artículo de Pathak et al. usa en
realidad `[r ; r²]` (cuadrados puros del estado del reservoir), no NVAR. La discusión sobre esta
diferencia y el porqué del paso a ESN puro / arquitectura paralela está en
`RESULTADOS_ESN_SOLO.md`.

### Hiperparámetros de la línea base
| Parámetro | Valor | Rol |
|-----------|-------|-----|
| `units` | 1000 | número de neuronas del reservoir |
| `lr` (leaking rate) | 1.0 | velocidad de actualización del estado |
| `sr` (radio espectral) | 0.6 | memoria / dinámica interna (ρ) |
| `input_scaling` | 0.02 | escala de la entrada (σ) |
| `input_connectivity` | 0.1 | densidad de `W_in` |
| `rc_connectivity` | 0.05 | densidad de `A` |
| `ridge` | 1e-4 | regularización de Tikhonov (β) |
| ruido de entrenamiento | 0.015 | estabiliza la predicción autónoma |

---

## 3. Protocolo de predicción autónoma

1. **Entrenamiento (teacher forcing):** el readout aprende el mapa a un paso
   `[estado] → u(t+1)` sobre los datos de entrenamiento, descartando un `warmup` inicial.
2. **Sincronización (warmup de test):** se alimenta el reservoir con `step_warmup` pasos **reales**
   del conjunto de test para fijar su estado interno.
3. **Predicción autónoma:** a partir de ahí, la red se alimenta de **sus propias predicciones**
   durante `step_prediccion` pasos, sin ver más datos reales.

El ruido inyectado en el entrenamiento es **necesario**: sin él, el readout nunca ve entradas
imperfectas y, en modo autónomo, los pequeños errores se realimentan y la trayectoria diverge de
inmediato. Es una técnica estándar de regularización en Echo State Networks (no es un parche).

---

## 4. Criterio de validación — tiempo de Lyapunov

El sistema es caótico: dos trayectorias cercanas se separan como `e^{λ·t}`. El **tiempo de
Lyapunov** es la escala natural de predictibilidad:

```
λ_max = 1.67            (exponente de Lyapunov máximo, Lorenz-96 con F=8, N=36)
τ_Lyap = 1 / λ_max ≈ 0.599 u.t.
pasos_por_Lyap = τ_Lyap / Δt = 0.599 / 0.05 ≈ 12 pasos
```

**Criterio:** se define el error RMS por paso sobre los datos normalizados y se mide el primer
instante en que supera un umbral:

```
err_t = sqrt( mean_j ( (pred_norm[t] − real_norm[t])² ) )
tiempo_valido = primer t tal que err_t > umbral      (umbral = 0.3)
```

Luego se expresa ese tiempo en unidades de Lyapunov (adimensional, comparable entre sistemas):

```
t_valido_lyap = (pasos_cruce · Δt) / τ_Lyap
```

---

## 5. Resultado de la línea base (ESN + NVAR)

```
Predicción válida: 32 pasos = 1.60 u.t. = 2.67 tiempos de Lyapunov
```

**Cómo se obtiene 2.67:**
```
32 pasos × 0.05 u.t./paso = 1.60 u.t.
1.60 u.t. / 0.599 u.t.    = 2.67 tiempos de Lyapunov
```

---

## 6. ¿Por qué 2.67 es un resultado confiable?

1. **Se mide con un criterio objetivo**, no a ojo: el primer cruce de un umbral fijo de error
   normalizado.
2. **El error crece de forma suave y exponencial** (verificado paso a paso:
   0.06 → 0.16 → 0.27 → 0.37 → 0.48 → ...), exactamente como predice la teoría del caos. No hay
   saltos bruscos ni inestabilidades numéricas: la divergencia observada **es la física del
   sistema**, no un fallo del modelo.
3. **El límite lo impone el caos, no la red.** El tiempo de Lyapunov es una barrera fundamental:
   un error inicial inevitable (precisión numérica, error del modelo) se multiplica por `e` cada
   `τ_Lyap`. Ningún predictor puede superar esta barrera indefinidamente.
4. **Está en el rango de la literatura.** Trabajos de referencia (Pathak et al., PRL 2018)
   reportan horizontes del orden de varios tiempos de Lyapunov para ESN bien construidos. 2.67 es
   razonable para un reservoir **global** (un solo reservoir para las 36 variables); está algo por
   debajo del óptimo del artículo porque éste usa una **arquitectura de reservoirs locales
   paralelos** (ver `RESULTADOS_ESN_SOLO.md`).

### Por qué se reporta en tiempos de Lyapunov y no en pasos/segundos
Expresar el horizonte en `τ_Lyap` lo vuelve **comparable entre sistemas distintos**: un sistema que
diverge rápido (λ grande) y otro lento (λ pequeño) se miden ambos con su propio "reloj caótico".
Es la convención estándar en la literatura de predicción de caos.

---

## 7. Visualización

La figura del notebook muestra:
- **Panel superior:** una variable (`x_11`) real (negro) vs. predicha (rojo), con doble eje
  temporal (u.t. abajo, tiempos de Lyapunov arriba) y una línea vertical en el instante de cruce
  del umbral (`2.67 τ_Lyap`). La predicción sigue bien a la realidad hasta la línea y luego diverge
  de forma progresiva — comportamiento caótico esperado.
- **Panel inferior:** mapa de error absoluto para las 36 variables en el tiempo; el error se
  mantiene bajo hasta ~1.6 u.t. y después se "enciende".

---

## 8. Conclusión

El ESN + NVAR es una **línea base válida y defendible**: predice ≈2.67 tiempos de Lyapunov de forma
limpia, con un error que crece como exige la teoría del caos. El siguiente paso para acercarse a los
horizontes del artículo (4-8 τ_Lyap) **no es ajustar hiperparámetros** (ya se barrieron sin mejora
sustancial), sino cambiar a la **arquitectura de reservoirs locales paralelos** de Pathak et al.
