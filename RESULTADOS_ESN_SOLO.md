# Por qué el ESN "puro" no alcanza el horizonte del artículo

**Contexto:** tras validar la línea base ESN + NVAR (`RESULTADOS_ESN_NVAR.md`, 2.67 tiempos de
Lyapunov), se intentó usar el método **fiel al artículo de Pathak et al.**, que emplea Echo State
Networks sin NVAR. Este documento resume qué se probó, qué resultados dieron los barridos de
hiperparámetros, y la conclusión de por qué un reservoir **único/global** no basta.

---

## 1. Las tres variantes de readout (aclaración importante)

| Variante | Qué entra al readout | ¿Fiel al artículo? |
|----------|----------------------|--------------------|
| ESN lineal puro | `r` (solo el estado del reservoir) | No (demasiado simple para Lorenz-96) |
| ESN + NVAR | `r` **+** términos cuadráticos de la **entrada** (incluye productos cruzados) | No (NVAR añade cruzados; el paper no) |
| **ESN con `[r ; r²]`** | `r` **+** cuadrados **puros del estado** (sin cruzados) | **Sí** (Pathak et al.) |

El artículo entrena:
```
v = W_out · [r ; r²]      con   r² = (r_1², r_2², ..., r_N²)   (cuadrados elemento a elemento)
```
No incluye productos cruzados `r_i·r_j`. Por eso **ni el ESN lineal puro ni NVAR** reproducen el
método; la variante correcta es `[r ; r²]`, construida manualmente (correr el reservoir, concatenar
`[R, R²]`, entrenar un Ridge).

---

## 2. Qué pasó al quitar NVAR

### 2.1 ESN lineal puro (`reservoir >> readout`)
La predicción autónoma **se aplanó**: la salida convergía a una constante (≈ la media del sistema) a
los pocos pasos. Horizonte ≈ **0.25 tiempos de Lyapunov**. Causa: sin la no linealidad extra, el
reservoir solo no tiene capacidad para representar la dinámica de Lorenz-96 y se relaja al punto
fijo. **Conclusión:** el ESN lineal puro es *más simple que* el artículo; no sirve para este sistema.

### 2.2 ESN con `[r ; r²]` (la variante del artículo)
El entrenamiento a un paso quedó **excelente** (NRMSE a 1 paso ≈ 0.004–0.07, según hiperparámetros),
pero la predicción **autónoma** divergía en pocos pasos. Esto motivó un diagnóstico sistemático.

---

## 3. Diagnóstico: descartar causas una por una

Se verificó, con mediciones directas, que **todo lo individual estaba bien**:

| Hipótesis | Prueba | Resultado |
|-----------|--------|-----------|
| Datos mal escalados | estadísticos de `U` | ✅ correctos (valores canónicos F=8) |
| Entrenamiento roto | NRMSE 1-paso en test | ✅ bajo (≈0.07) → el readout generaliza |
| Sobreajuste (pesos enormes) | `|W_out|` máx / medio | ✅ pequeños (máx ≈ 9, medio ≈ 0.33) |
| El estado no se arrastra en el bucle | comparar run paso-a-paso vs. de corrido | ✅ idénticos (diferencia 0.0) |
| Error salta de golpe (bug de reinyección) | curva de error autónomo paso a paso | ✅ crece **suave y exponencial** |

La curva de error autónomo (con la mejor configuración) crecía así:
```
paso:   1     2     3     4     5     6     7     8     9    ...
error: 0.06  0.16  0.27  0.37  0.48  0.60  0.73  0.88  1.02 ...
```
Crecimiento exponencial limpio ⇒ **no hay bug**; el modelo funciona, solo que su error inicial por
paso (~0.06) es suficientemente alto como para que el caos lo amplifique y cruce el umbral pronto.

---

## 4. Barridos de hiperparámetros

Se midió el horizonte con **validación robusta**: umbral 0.4 (estándar) y promedio sobre **8
ventanas** de test con distintos puntos de arranque (para que el número no dependa de una sola
ventana con suerte).

### 4.1 Barrido `units × lr` (sr=0.6, ridge=1e-6, ruido=0.01)

| units | lr | err 1-paso | horizonte (τ_Lyap) |
|------:|---:|-----------:|-------------------:|
| 2000 | 0.3 | 0.4329 | 0.11 ± 0.04 |
| 2000 | 0.5 | 0.1946 | 0.22 ± 0.04 |
| 2000 | 0.7 | 0.1182 | 0.34 ± 0.05 |
| 4000 | 0.3 | 0.6229 | 0.08 ± 0.00 |
| 4000 | 0.5 | 0.3452 | 0.18 ± 0.05 |
| 4000 | 0.7 | 0.2126 | 0.31 ± 0.08 |

Observaciones clave:
- **`lr` más alto es mejor** (contra la intuición inicial de bajarlo).
- **`units=4000` es PEOR que `units=2000`** en todas las filas. Más neuronas no ayuda → señal de
  límite **estructural**, no de capacidad.

### 4.2 Barrido `lr × sr × ruido` (units=2000, ridge=1e-7)

| lr | sr | ruido | err 1-paso | horizonte (τ_Lyap) |
|---:|---:|------:|-----------:|-------------------:|
| 0.7 | 0.5 | 0.000 | 0.1024 | 0.33 ± 0.06 |
| 0.7 | 0.5 | 0.005 | 0.1037 | 0.39 ± 0.07 |
| 0.9 | 0.5 | 0.005 | 0.0726 | 0.51 ± 0.07 |
| **1.0** | **0.5** | **0.005** | **0.0642** | **0.57 ± 0.09** |
| 1.0 | 0.9 | 0.000 | 0.0953 | 0.25 ± 0.00 |
| 1.0 | 1.3 | 0.000 | 0.2492 | 0.17 ± 0.00 |

Observaciones:
- Mejor combinación: `lr=1.0`, `sr=0.5`, ruido `0.005` → **0.57 τ_Lyap**.
- `sr` bajo gana; `sr=1.3` siempre malo.
- El error a un paso **no baja de ~0.06**, sin importar la configuración.

---

## 5. Conclusión: el techo es la arquitectura, no los hiperparámetros

Tras barrer `units`, `lr`, `sr`, `ridge`, `input_scaling` y ruido (decenas de combinaciones), el
horizonte de un **reservoir único/global** topa en **~0.57 tiempos de Lyapunov**, y el error a un
paso no baja de ~0.06. Añadir neuronas incluso empeora.

**Por qué un reservoir único no puede más:** Lorenz-96 tiene **localidad espacial** — cada variable
`x_j` solo interactúa con sus vecinas (`x_{j−2}, x_{j−1}, x_{j+1}`). Un reservoir global intenta
aprender las 36 dinámicas acopladas a la vez, mezclando todo con todo, y se satura. No tiene la
capacidad de modelar con precisión las 36 variables simultáneamente; por eso más neuronas (4000) no
mejora: es el mismo problema global con más parámetros.

**Lo que hace el artículo de Pathak et al.** para llegar a 4-8 τ_Lyap: **no usa un reservoir único**,
sino **reservoirs locales paralelos**:
- divide las variables en grupos (p. ej. 6 grupos de 6);
- cada grupo tiene su propio reservoir pequeño;
- cada reservoir ve su grupo **más algunas variables vecinas de solapamiento** (para capturar el
  acoplamiento local);
- cada uno predice solo su grupo y luego se ensamblan.

Esto explota la localidad de Lorenz-96 y es la razón por la que el artículo alcanza horizontes
largos. Es un **cambio de arquitectura**, que es el siguiente paso del proyecto (ESN paralelo).

---

## 6. Resumen comparativo

| Enfoque | Horizonte (τ_Lyap) | Veredicto |
|---------|-------------------:|-----------|
| ESN + NVAR (línea base) | **2.67** | Válido y defendible como baseline |
| ESN lineal puro | 0.25 | Insuficiente (se aplana) |
| ESN `[r ; r²]`, un reservoir global | ≤ 0.57 | Techo estructural, no de ajuste |
| ESN `[r ; r²]`, reservoirs locales paralelos | (en desarrollo) | Enfoque fiel al artículo, objetivo 4-8 |
