# Planteamiento del Proyecto — Predicción de Sistemas Caóticos con Reservoir Computing

**Materia:** Herramientas Computacionales
**Tema:** Predicción "model-free" (sin conocer las ecuaciones) de sistemas caóticos usando
*reservoir computing* en Python, con validación en un sistema simulado y aplicación a datos reales.
**Artículo de referencia:** J. Pathak, B. Hunt, M. Girvan, Z. Lu y E. Ott,
*"Model-Free Prediction of Large Spatiotemporally Chaotic Systems from Data: A Reservoir
Computing Approach"*, **Phys. Rev. Lett. 120, 024102 (2018)**, DOI: 10.1103/PhysRevLett.120.024102.

> Este documento reemplaza el planteamiento anterior (motor de Carnot browniano).
> Contiene el método **paso por paso con todas las ecuaciones**, sin código.

---

## 0. Notación y convenciones

| Símbolo | Significado | Dimensión |
|---------|-------------|-----------|
| `u(t)` | vector de entrada (estado observado del sistema) en el instante `t` | `D_in` |
| `v(t)` | vector de salida (predicción del reservoir) | `D_out` |
| `r(t)` | estado interno del reservoir (las "neuronas") | `D_r` |
| `W_in` | matriz de entrada → reservoir (FIJA, aleatoria) | `D_r × D_in` |
| `A` | matriz de adyacencia del reservoir (FIJA, aleatoria) | `D_r × D_r` |
| `W_out` | matriz de salida reservoir → predicción (SE ENTRENA) | `D_out × 2·D_r` |
| `Δt` | paso de tiempo | escalar |
| `ρ` | radio espectral de `A` (mayor autovalor en módulo) | escalar |
| `σ` | escala de los pesos de entrada `W_in` | escalar |
| `κ` | grado medio de la red del reservoir (conexiones por nodo) | escalar |
| `β` | parámetro de regularización de Tikhonov (ridge) | escalar |
| `Λ` | exponente de Lyapunov máximo del sistema | escalar |

Convención de tiempo: discreto, `t = 0, Δt, 2Δt, …`.
Parámetros de referencia del paper: `D_r = 5000`, `ρ = 0.6`, `σ = 1.0`, `κ = 3`, buffer `l = 6`.

---

## 1. El problema (qué queremos lograr)

Un sistema caótico es **determinista** pero **sensible a las condiciones iniciales**: dos estados que
difieren en `d_0` se separan como

```
d(t) ≈ d_0 · e^(Λ t)                                   (Ec. 1)
```

donde `Λ` es el **exponente de Lyapunov máximo**. El **tiempo de Lyapunov** `t_Λ = 1/Λ` es el tiempo en
que el error se multiplica por `e`. **Todos los errores de predicción se reportarán en unidades de
`Λ·t`** (como en las figuras del paper), no en segundos.

**Objetivo:** dado un histórico de observaciones `u(t)` en `−T ≤ t ≤ 0`, predecir `u(t)` para `t > 0`
**sin usar las ecuaciones** del sistema, solo con los datos.

---

## 2. El reservoir computer (las 3 piezas)

```
 u(t) ──► [W_in] ──► ( RESERVOIR: A, r(t) ) ──► [W_out] ──► v(t)
          FIJA        FIJA                       SE ENTRENA
```

- `W_in` y `A` se generan **una vez, al azar, y NO se entrenan**.
- `W_out` es **lo único que se entrena**, mediante una regresión lineal.

### 2.1 Construcción de las matrices fijas (paso previo)

- **`W_in`**: cada elemento se toma de una distribución uniforme en `[−σ, σ]`.
- **`A`**: red dirigida de Erdős–Rényi de grado medio `κ` (cada nodo conecta en promedio con `κ`
  nodos). Luego se **reescala** para fijar su radio espectral:

```
A ← A · ( ρ / |λ_max(A)| )                            (Ec. 2)
```

  donde `λ_max(A)` es el autovalor de mayor módulo de `A` antes de reescalar. Esto garantiza la
  **"echo state property"** (el reservoir recuerda lo reciente y olvida lo viejo). Requiere `ρ < 1`.

### 2.2 Dinámica del reservoir (cómo evoluciona el estado interno)

En cada paso de tiempo el estado se actualiza con (Ec. 1 del paper):

```
r(t + Δt) = tanh( A·r(t) + W_in·u(t) )                (Ec. 3)
```

`tanh` se aplica componente a componente; acota el estado y aporta la no-linealidad.

### 2.3 Lectura de salida (cómo se produce la predicción)

```
v(t) = W_out · [ r(t) ; r(t)² ]                       (Ec. 4)
```

donde `[r ; r²]` es el vector que concatena `r(t)` con su cuadrado componente a componente
(dimensión `2·D_r`).

> **Detalle crítico del paper (nota [16]):** usar solo `r` (sin `r²`) **falla** por la simetría impar
> de `tanh`. El término `r²` rompe esa simetría y es necesario. HAY que incluirlo.

---

## 3. Fase de entrenamiento (ajustar `W_out`)

Se usa el tramo conocido `−T ≤ t ≤ 0` (Fig. 1a del paper).

1. **Recolectar estados:** inyectar los datos reales `u(t)` uno a uno en la Ec. (3) y **guardar** cada
   estado `r(t)` resultante. Descartar un transitorio inicial (los primeros pasos) para que se cumpla
   la echo state property.
2. **Construir el vector de características:** `X(t) = [ r(t) ; r(t)² ]` para cada `t`.
3. **Objetivo de predicción:** queremos que la salida sea el **siguiente** valor del sistema,
   es decir el dato deseado es `v_d(t) = u(t + Δt)`.
4. **Entrenar `W_out`** resolviendo la regresión lineal de mínimos cuadrados con regularización de
   Tikhonov (ridge):

```
W_out = argmin_W  Σ_t || W·X(t) − v_d(t) ||²  +  β·||W||²      (Ec. 5)
```

cuya solución cerrada es:

```
W_out = V_d · Xᵀ · ( X·Xᵀ + β·I )⁻¹                   (Ec. 6)
```

donde `X` apila por columnas los `X(t)`, `V_d` apila los `v_d(t)`, `I` es la identidad y `β` controla
el sobreajuste. **Esto es toda la "parte de aprendizaje": una sola regresión, sin backpropagation.**

---

## 4. Fase de predicción autónoma (pronosticar el futuro)

Para `t > 0` ya no hay datos reales. Se **realimenta la salida** (Fig. 1b del paper): la predicción se
convierte en la entrada del paso siguiente.

```
r(t + Δt) = tanh( A·r(t) + W_in·v(t) )                (Ec. 7)
v(t + Δt) = W_out · [ r(t+Δt) ; r(t+Δt)² ]            (Ec. 8)
```

(Nótese que en la Ec. 7 la entrada es `v(t)` —la propia predicción— en lugar del dato real `u(t)`.)
El reservoir queda **autónomo** y genera su propia trayectoria futura.

**Reinicialización para predecir desde un instante `θ`:** poner `r = 0` y alimentar el reservoir con
datos **reales** durante un corto periodo `[θ − ε, θ]` (en el paper `ε = 10` pasos) usando la Ec. (3);
esto fija un `r(θ)` coherente, y a partir de ahí se predice con las Ec. (7)–(8).

---

## 5. Esquema de reservoirs en PARALELO (la contribución del paper)

Para sistemas espaciales grandes, un solo reservoir no escala. Solución: dividir el espacio y usar
**un reservoir por región local**.

- El sistema tiene `Q` variables espaciales: `u(t) = (u_1, …, u_Q)`.
- Se reparten en `g` grupos contiguos de `q` variables cada uno (`g·q = Q`).
- Cada reservoir `R_i` predice su grupo `g_i`, pero **recibe también un buffer** de `l` puntos vecinos a
  izquierda y derecha. Su entrada es:

```
h_i(t) = ( u_{(i−1)q − l + 1}, … , u_{iq + l} )       (Ec. 9)
```

(los índices se toman módulo `Q`, condiciones de frontera periódicas).

- Entrenamiento en paralelo (análogo a la Ec. 3), uno por reservoir:

```
r_i(t + Δt) = tanh( A_i·r_i(t) + W_in,i·h_i(t) )      (Ec. 10)
```

- Cada reservoir aprende su propia `W_out,i` con la regresión de la Sección 3, para que
  `W_out,i·[r_i ; r_i²] ≈ g_i(t)` (solo su grupo).
- En predicción, las salidas de los reservoirs se concatenan para reconstruir el estado completo, y los
  buffers `h̃_i` se arman con las predicciones de los vecinos.

> **Simplificación con simetría:** si el sistema es homogéneo (simetría de traslación), basta entrenar
> **un** reservoir y copiar sus pesos a todos (`W_in,i = W_in,1`, `A_i = A_1`, `W_out,i = W_out,1`).
> Si hay inhomogeneidad, cada reservoir se entrena por separado. (Figs. 6a vs 6b del paper.)

---

## 6. Métricas de evaluación

### 6.1 Error cuadrático medio (RMSE) en el tiempo

Entre la predicción `ṽ(t)` y la verdad `u(t)`:

```
RMSE(t) = sqrt( (1/Q) · Σ_j ( ṽ_j(t) − u_j(t) )² )    (Ec. 11)
```

Se promedia sobre `K` intervalos de predicción independientes y sobre varias realizaciones aleatorias
del reservoir (el paper: `K = 30`, 10 realizaciones). **El eje de tiempo se expresa como `Λ·t`.**

### 6.2 Tiempo de predicción válido

El tiempo (en unidades de Lyapunov) hasta que el `RMSE(t)` supera un umbral fijo (p. ej. 0.3–0.5). Es
el número que resume "cuánto predice bien el reservoir". Meta de referencia del paper: **~8 tiempos de
Lyapunov**.

### 6.3 Exponente de Lyapunov `Λ` (necesario para las dos métricas)

Para el sistema simulado se estima con el método estándar: integrar la trayectoria y una perturbación
`δ(t)` en paralelo, renormalizando periódicamente, y promediar el crecimiento logarítmico:

```
Λ = lim_{N→∞} (1 / (N·Δt)) · Σ_{n=1}^{N} ln( |δ(nΔt⁻)| / |δ((n−1)Δt)| )    (Ec. 12)
```

---

## 7. Fuente de datos: sistema simulado + datos reales (enfoque híbrido)

### 7.1 Sistema simulado de validación — Lorenz-96

Sistema caótico espacialmente extendido (ideal para el esquema paralelo). Para `j = 1, …, Q` con
frontera periódica:

```
dx_j/dt = ( x_{j+1} − x_{j−2} ) · x_{j−1} − x_j + F    (Ec. 13)
```

Con forzamiento `F = 8` el sistema es caótico. Ventaja: conocemos la **verdad de referencia** y podemos
calcular `Λ` (Ec. 12) → validación rigurosa, reproducción de las figuras del paper.

> Alternativa fiel al paper: ecuación de **Kuramoto–Sivashinsky**
> `y_t = −y·y_x − y_xx − y_xxxx (+ μ·cos(2πx/λ))`. Es más costosa de integrar; Lorenz-96 se recomienda
> para el curso por simplicidad equivalente en lo conceptual.

### 7.2 Datos reales — caso de estudio

El **método es idéntico**; solo cambia el origen de `u(t)`. Candidatos:
- **Clima solar:** número de manchas solares (SILSO) o flujo solar F10.7 (series largas, mensuales/diarias).
- **Contaminación:** PM2.5 por hora de estaciones de monitoreo (p. ej. OpenAQ).

**Pipeline con datos reales:**
1. Descargar la serie. 2. Limpiar (huecos, outliers). 3. Normalizar (media 0, desviación 1).
4. Partir 80% entrenamiento / 20% evaluación. 5. Entrenar (Sección 3). 6. Predecir (Sección 4).
7. Comparar con el 20% apartado.

**Diferencias a tener presentes:** en datos reales no hay "verdad de referencia" perfecta (hay ruido de
medición), normalmente es 1 sola serie (no aprovecha el esquema paralelo salvo multi-estación), y `Λ`
es difícil de estimar. Por eso Lorenz-96 aporta el rigor y los datos reales el caso del mundo real.

---

## 8. Capas de profundidad (para que NO sea un proyecto simple)

1. Medir todo en **tiempos de Lyapunov** (Ec. 1, 11, 12), no en segundos.
2. Comparar **reservoir único vs. reservoir paralelo** y mostrar la escalabilidad con el tamaño `Q`/`L`.
3. **Barrido de hiperparámetros** `ρ, σ, κ, D_r, β` con mapas de desempeño.
4. Incluir el término **`r²`** en la salida (Ec. 4) y comparar contra usar solo `r`.
5. **Baselines de comparación:** persistencia (`v(t+Δt) = u(t)`) y opcionalmente una RNN/LSTM, para
   mostrar que el reservoir es competitivo y mucho más barato de entrenar.
6. **Caso de datos reales** (Sección 7.2) como capa final.

---

## 9. Flujo global del método (resumen en pasos)

```
(1) Elegir sistema → generar u(t) de Lorenz-96 (Ec. 13)  [o cargar datos reales]
(2) Estimar Λ del sistema (Ec. 12)                        → para medir en tiempos de Lyapunov
(3) Construir reservoir: W_in (uniforme ±σ), A (Erdős–Rényi), reescalar a ρ (Ec. 2)
(4) ENTRENAR:  evolucionar r(t) (Ec. 3), guardar estados, resolver W_out (Ec. 6)
(5) PREDECIR:  realimentar la salida (Ec. 7–8) de forma autónoma
(6) (opcional) PARALELIZAR por regiones locales (Ec. 9–10)
(7) EVALUAR:  RMSE(t) en Λ·t (Ec. 11), tiempo de predicción válido, único vs paralelo, baselines
(8) APLICAR el mismo método a una serie real (manchas solares / PM2.5)
```

---

## Referencia

Pathak, J., Hunt, B., Girvan, M., Lu, Z. & Ott, E. *Model-Free Prediction of Large Spatiotemporally
Chaotic Systems from Data: A Reservoir Computing Approach.* Phys. Rev. Lett. **120**, 024102 (2018).
DOI: 10.1103/PhysRevLett.120.024102.
