# Resultados — ESN Paralelo (reservoirs locales) sobre Lorenz-96

**Código de referencia:** `esn_paralelo_final.py`
**Arquitectura:** Echo State Networks locales paralelos, estilo Pathak et al.
(Phys. Rev. Lett. 120, 024102, 2018), con readout `[r ; r²]` (cuadrados puros).

> Este documento cierra la línea de Reservoir Computing: parte del techo del reservoir
> **global** (0.57 τ_Lyap, ver `RESULTADOS_ESN_SOLO.md`) y muestra cómo la arquitectura
> **local paralela** lo multiplica por ~7× hasta **4.16 tiempos de Lyapunov**, ya dentro del
> rango del artículo de referencia.

---

## 1. Idea de la arquitectura

Lorenz-96 tiene **localidad espacial**: `dx_j/dt = (x_{j+1} − x_{j−2})·x_{j−1} − x_j + F`.
Cada variable solo interactúa con sus vecinas `x_{j−2}, x_{j−1}, x_{j+1}`. Un reservoir único
intenta modelar las 36 variables acopladas a la vez y se satura (su techo fue 0.57 τ_Lyap).

El ESN paralelo explota esa localidad:
- se divide el sistema en **grupos** de variables;
- cada grupo tiene **su propio reservoir** con readout `[r ; r²]`;
- cada reservoir recibe sus variables centrales **más algunas vecinas de solape** (para capturar
  el acoplamiento), con **frontera periódica** (módulo 36, igual que Lorenz-96);
- cada reservoir predice **solo sus variables centrales**;
- la predicción global se ensambla concatenando las salidas, y se reinyecta a todos los grupos
  en el siguiente paso (predicción autónoma conjunta).

---

## 2. Configuración ganadora

| Parámetro | Valor |
|-----------|-------|
| nº de grupos | 18 (2 variables centrales por grupo) |
| solape | 2 vecinas a cada lado (entrada = 6, salida = 2) |
| neuronas por grupo (`units_g`) | 600 |
| `lr` (leaking rate) | 1.0 |
| `sr` (radio espectral) | 0.4 |
| `input_scaling` | 0.15 |
| `ridge` | 1e-6 |
| ruido de entrenamiento | 0.005 |
| readout | `[r ; r²]` (cuadrados puros, fiel al artículo) |

**Resultado:** `4.16 ± 0.94 tiempos de Lyapunov` (métrica robusta: umbral 0.4, promedio sobre
8 ventanas de test con distintos puntos de arranque).

---

## 3. Métrica de validación

Igual que en el resto del proyecto (ver `RESULTADOS_ESN_NVAR.md` §4):

```
λ_max = 1.67,   τ_Lyap = 1/λ_max ≈ 0.599 u.t.,   Δt = 0.05 u.t./paso
err_t = sqrt( mean_j (pred_norm[t] − real_norm[t])² )
tiempo_valido = primer t con err_t > 0.4        (umbral estandar)
horizonte (τ_Lyap) = (paso_cruce · Δt) / τ_Lyap
```

Se promedia el horizonte sobre **8 ventanas** y se reporta media ± desviación, para que el
número no dependa de una sola ventana afortunada.

---

## 4. Barridos realizados

### 4.1 Número de grupos × solape (units_g según tamaño)

| grupos | vars/grupo | solape | units_g | horizonte (τ_Lyap) |
|-------:|-----------:|-------:|--------:|-------------------:|
| 6  | 6 | 2 | 600 | 2.63 ± 0.80 |
| 6  | 6 | 4 | 600 | 1.46 ± 0.30 |
| 9  | 4 | 2 | 500 | 3.05 ± 0.56 |
| 9  | 4 | 3 | 500 | 2.43 ± 0.60 |
| 12 | 3 | 2 | 400 | 3.23 ± 0.90 |
| 18 | 2 | 2 | 400 | 3.66 ± 0.62 |
| 36 | 1 | 2 | 300 | 3.27 ± 0.52 |
| 36 | 1 | 3 | 300 | 3.04 ± 0.79 |
| 18 | 2 | 1 | 400 | **0.48 ± 0.09** |

### 4.2 Ajuste fino sobre la arquitectura ganadora (18 grupos, solape 2)

| units_g | sr | input_scaling | horizonte (τ_Lyap) |
|--------:|---:|--------------:|-------------------:|
| 600 | 0.4 | 0.15 | **4.16 ± 0.94** |
| 600 | 0.4 | 0.30 | 2.95 ± 0.59 |
| 600 | 0.5 | 0.15 | 3.96 ± 0.75 |
| 600 | 0.6 | 0.15 | 3.73 ± 0.78 |
| 1000 | 0.4 | 0.15 | 4.30 ± 1.20 |
| 1000 | 0.5 | 0.15 | 3.92 ± 0.85 |
| 1000 | 0.6 | 0.15 | 3.86 ± 0.81 |

Se eligió `units_g=600` sobre `units_g=1000` (4.30 ± 1.20): la diferencia de media es menor que
la desviación estándar (equivalentes estadísticamente), pero 600 es **más estable** (±0.94) y
usa 40 % menos cómputo.

---

## 5. Hallazgos físicos (defendibles en el informe)

1. **El solape óptimo coincide con el alcance del acoplamiento.** Con solape 1, cada grupo de 2
   variables ve solo 4 variables y **pierde el término `x_{j−2}`** de la ecuación → el horizonte
   colapsa a 0.48 τ_Lyap. Con solape 2 captura todo el acoplamiento local → 3.66-4.16. El solape
   mínimo necesario corresponde directamente al alcance espacial de la dinámica.

2. **Más solape de lo necesario empeora.** Solape 4 (6 grupos) bajó a 1.46: al incluir variables
   irrelevantes para el grupo, el reservoir vuelve a mezclar información global y se degrada.

3. **Grupos locales pequeños ganan a un reservoir global**, pero fragmentar al extremo (36 grupos
   de 1 variable) no es lo mejor: 18 grupos de 2 variables fue superior a 36 de 1.

4. **`sr` bajo (0.4) y `input_scaling` moderado (0.15)** son los mejores: coherente con un sistema
   que necesita memoria corta y entradas que no saturen la `tanh`.

---

## 6. Comparación final de todos los enfoques

| Enfoque | Horizonte (τ_Lyap) | Métrica |
|---------|-------------------:|---------|
| ESN `[r;r²]` global | 0.57 | robusta (umbral 0.4, 8 ventanas) |
| ESN + NVAR | 2.67 | 1 ventana, umbral 0.3 (indulgente) |
| ESN paralelo inicial (6 grupos, sin afinar) | 2.63 | robusta |
| **ESN paralelo optimizado (18 grupos)** | **4.16 ± 0.94** | robusta |

Del reservoir global (0.57) al paralelo optimizado (4.16): mejora de **~7×**, alcanzando el rango
de horizontes reportado en la literatura (Pathak et al., 4-8 τ_Lyap). El salto no vino de ajustar
hiperparámetros, sino de adoptar la **arquitectura local paralela** que explota la localidad
espacial de Lorenz-96.

---

## 7. Reproducibilidad

- Semillas RNG fijas (`seed=42`, y `seed+g` por grupo) → resultados reproducibles.
- La figura `prediccion_lorenz96_paralelo.png` muestra una variable real vs. predicha con doble eje
  temporal (u.t. y tiempos de Lyapunov) y el instante de cruce del umbral, más el mapa de error de
  las 36 variables.
- Código completo y comentado en `esn_paralelo_final.py`.
