# Repartición del Proyecto — Motor de Carnot Browniano

**Materia:** Herramientas Computacionales
**Proyecto:** Simulación de un motor térmico browniano mediante ecuaciones de Langevin
**Artículo de referencia:** I. A. Martínez, É. Roldán, L. Dinis, D. Petrov, J. M. R. Parrondo y
R. A. Rica, *"Brownian Carnot engine"*, **Nature Physics** (2015), DOI: 10.1038/nphys3518.
**Integrantes:** 5
**Modalidad:** 5 sesiones (clases), un segmento por integrante

---

## 1. Objetivo del proyecto

**Reproducir computacionalmente** el experimento del artículo de Martínez et al. (2015): un
**motor de Carnot browniano** formado por una única partícula coloidal atrapada en una pinza óptica
(potencial armónico de rigidez `κ`), sometida a un ciclo de Carnot entre dos baños térmicos a
temperaturas `T_c` y `T_h`. En vez de medir en el laboratorio, **simulamos** la dinámica con la
ecuación de Langevin y calculamos, con las herramientas de la **termodinámica estocástica**, el
trabajo, el calor, la eficiencia y la potencia del motor, incluyendo sus **fluctuaciones**.

La meta de validación es **reproducir las figuras 1c, 2 y 3 del artículo** con los parámetros exactos
que el propio paper da en sus Métodos.

### Modelo físico (del artículo)

- Potencial óptico armónico: `U(x,t) = κ(t)·x² / 2`.
- Hamiltoniano (CON inercia): `H = κ x² / 2 + p² / (2m)`, con `p = m·(dx/dt)`.
- Fuerza conjugada al parámetro de control: `F_κ = ∂H/∂κ = x² / 2`.
- Dinámica: **ecuación de Langevin con inercia** (subamortiguada)
  ```
  m (d²x/dt²) = -κ(t) x - γ (dx/dt) + √(2 γ k_B T(t)) ξ(t)
  ```
  con `γ` fricción, `k_B` constante de Boltzmann y `ξ(t)` ruido blanco gaussiano.

> **Cambio de alcance respecto a la primera propuesta:** el artículo usa el modelo **con inercia**
> (porque el calor de Sekimoto incluye el cambio de energía cinética `dE_cin`). Por eso NO basta el
> régimen sobreamortiguado: se integra el Langevin completo con un esquema **BAOAB**, que es estable y
> muestrea bien la distribución de Boltzmann en un potencial armónico.

---

## 2. Parámetros EXACTOS del artículo (para reproducir — Métodos del paper)

Estos valores son el acuerdo numérico del grupo. Los define el artículo:

| Parámetro | Símbolo | Valor |
|-----------|---------|-------|
| Temperatura fría (isotermas 1 y 2) | `T_c` | **300 K** |
| Temperatura caliente (isotermas 3 y 4) | `T_h` | **525 K** |
| Rigidez inicial | `κ_I = κ(0)` | **2.0 pN·µm⁻¹** |
| Rigidez máxima | `κ_III = κ(τ/2)` | **20.0 pN·µm⁻¹** |
| Rigidez de referencia | `κ_II = κ(τ/4)` | **6.5 pN·µm⁻¹** |
| Rigidez `κ_IV` (impuesta por geometría) | `κ_IV = (κ_III/κ_II)·κ_I` | **6.2 pN·µm⁻¹** |
| Eficiencia de Carnot | `η_C = 1 − T_c/T_h` | **≈ 0.43** |
| Eficiencia Curzon–Ahlborn | `η_CA = 1 − √(T_c/T_h)` | **≈ 0.25** |
| Potencia máxima (resultado a reproducir) | `P_max` | **≈ 6.34 k_B T_c / s** |
| Tiempos de ciclo | `τ` | de **10 ms** a **200 ms** |
| Paso de muestreo | `Δt` | **5 ms** (2 kHz) — usar `dt` menor para integrar |
| Tiempo de la "serie experimental" | `τ_exp` | **50 s** (fija cuántos ciclos se promedian) |
| Partícula | diámetro / medio | esfera de poliestireno **1 µm** en agua |

### Protocolo del ciclo (Métodos del paper)

- **Rigidez simétrica en el tiempo:** `κ(t) = κ(τ − t)`. Crece **cuadráticamente** de `t=0` a `t=τ/2`
  y decrece al mismo ritmo de `τ/2` a `τ`.
- **Tiempo de cruce adiabático:**
  `τ* = τ − (τ/2)·√[(κ_III/κ_II − 1)/(κ_III/κ_I − 1)] ≈ 0.76 τ`.
- **Temperaturas por tramo:** `T_I = T_II = T_c = 300 K` para `t ∈ [0, τ/2]`;
  `T_III = T_IV = T_h = 525 K` para `t ∈ [τ/2, τ*]`.
- **Adiabáticas:** `T` cambia suavemente manteniendo **`T²(t)/κ(t) = constante`** (conserva la
  entropía de Shannon → proceso adiabático efectivo).

### Los 4 tramos (Fig. 1a del paper)

```
(1) Compresión     (2) Compresión   (3) Expansión    (4) Expansión
    isotérmica  →      adiabática  →    isotérmica  →    adiabática  → (vuelve a 1)
    T = Tc             T: Tc→Th         T = Th           T: Th→Tc
```

---

## 3. Fórmulas de termodinámica estocástica (Métodos del paper)

Son las fórmulas que debe implementar el Segmento 4.

- **Trabajo** (integral de Stratonovich, con `F_κ = x²/2`):
  ```
  dW_t = ½ [ F_κ(x_t) + F_κ(x_{t+Δt}) ] · (κ_{t+Δt} − κ_t)
  ```
- **Calor** (Sekimoto, primera ley): `dQ_t = dH_t − dW_t = dU_t + dE_cin,t − dW_t`, con
  `dU_t = ½(κ_{t+Δt} x²_{t+Δt} − κ_t x²_t)` y `dE_cin,t = (m/2)(v²_{t+Δt} − v²_t)`.
- **Trabajo/calor por tramo:** suma de `dW_t` / `dQ_t` desde el inicio al fin del subproceso α (α=1..4).
- **Eficiencia estocástica** (ec. 1), sumando sobre `i` ciclos:
  ```
  η_τ^(i) = −W_τ^(i) / ( Q_{2,τ}^(i) + Q_{3,τ}^(i) + Q_{4,τ}^(i) )
  ```
  (El paper discute además `η₁`, `η₂`, `η₃` según qué calores se cuentan — ver Métodos.)
- **Potencia:** `P_τ = −⟨W_τ⟩ / τ`.
- **Leyes a verificar:** `⟨W_τ⟩ = ⟨W_∞⟩ + Σ_ss/τ` y `⟨Q_τ⟩ = ⟨Q_∞⟩ − Σ_ss/τ` (ajuste `A + B/τ`).

---

## 4. Stack tecnológico y arquitectura

**C++** para el núcleo de cómputo (integrar Langevin en muchos ciclos y realizaciones — es pesado),
**Python** para análisis y figuras (NumPy/Matplotlib/Pandas). Son las herramientas del curso.

| Capa | Lenguaje | Responsabilidad |
|------|----------|-----------------|
| Núcleo de simulación | **C++** | Integrador BAOAB, protocolo del ciclo, orquestación, escritura de datos |
| Análisis y figuras | **Python** | Termodinámica estocástica, eficiencia, PDFs, gráficas, informe |

**Frontera C++ → Python:** C++ escribe las trayectorias a **CSV**; Python las lee para analizar.

```
Motor-Carnot-Browniano/
├── src/                      # C++
│   ├── langevin.hpp/.cpp      # Seg. 1 (integrador BAOAB)
│   ├── ciclo.hpp/.cpp         # Seg. 2 (protocolo κ(t), T(t))
│   ├── simulacion.hpp/.cpp    # Seg. 3 (orquestador)
│   └── main.cpp               # Seg. 3
├── analisis/                 # Python
│   ├── termodinamica.py       # Seg. 4 (W, Q, η, P)
│   └── visualizacion.py       # Seg. 5 (Figs. 1c, 2, 3)
├── datos/                    # CSV generados por C++
├── figuras/                  # Figuras reproducidas
├── CMakeLists.txt             # Seg. 3
├── requirements.txt           # Seg. 5
└── README.md                  # Informe final (Seg. 5)
```

### Contrato del CSV (frontera C++ → Python) — acordar en Sesión 1

Columnas con encabezado, separadas por coma:

```
realizacion,tiempo,tramo,x,v,T,kappa
```

- `realizacion`: índice de corrida (0..N−1) para promediar el ruido.
- `tiempo`: tiempo de simulación. · `tramo`: 1=iso. comp., 2=adiab. comp., 3=iso. exp., 4=adiab. exp.
- `x`: posición. · `v`: velocidad (necesaria para `dE_cin`). · `T`: temperatura. · `kappa`: rigidez.

> Mientras una parte no esté lista, usar *stubs* que devuelvan datos de prueba. La interfaz es el acuerdo.

---

## 5. Repartición por integrante y por sesión

Cada integrante es **dueño** de un segmento. Al inicio de cada sesión: 10 min de sincronización.

---

### 👤 Integrante 1 — Integrador de Langevin con inercia (BAOAB) · C++

**Objetivo:** el corazón numérico que avanza la partícula (posición y velocidad) un paso de tiempo.

| Sesión | Tarea |
|--------|-------|
| 1 | Acuerdo de interfaces (§4). Crear `src/langevin.hpp/.cpp`. Generador de ruido gaussiano (`std::normal_distribution`) con semilla configurable. Unidades físicas (pN, µm, ms, K, `k_B`). |
| 2 | Implementar el **paso BAOAB** del Langevin con inercia (subpasos B-A-O-A-B) para `m`, `γ`, `κ`, `T` dados. Exponer `paso(Estado, κ, T, dt) -> Estado`. |
| 3 | **Validar con equipartición:** a `κ`, `T` fijos, comprobar `⟨x²⟩ = k_B T/κ` y `⟨v²⟩ = k_B T/m`. Documentar la prueba (criterio de éxito del integrador). |
| 4 | Estimar `γ` y `m` físicos (esfera de 1 µm en agua: `m` del poliestireno, `γ = 6πηr` de Stokes). Elegir `dt` estable (`dt ≪ Δt=5 ms`) y justificarlo. |
| 5 | Optimización (`-O2/-O3`), revisión de código con el grupo, apoyo al informe (sección "integrador"). |

**Partes:** unidades físicas · generador de ruido · paso BAOAB · validación por equipartición · parámetros `m`, `γ`, `dt`.

---

### 👤 Integrante 2 — Protocolo del ciclo de Carnot · C++

**Objetivo:** reproducir EXACTAMENTE el protocolo `κ(t)` y `T(t)` del paper (§2).

| Sesión | Tarea |
|--------|-------|
| 1 | Acuerdo de interfaces. Crear `src/ciclo.hpp/.cpp`. Cargar los parámetros exactos (`κ_I, κ_II, κ_III, κ_IV, T_c, T_h, τ, τ*`). |
| 2 | Implementar `κ(t)`: crecimiento **cuadrático** `[0, τ/2]`, decrecimiento simétrico `[τ/2, τ]`, con `κ(t)=κ(τ−t)`. Verificar que pasa por `κ_I, κ_II, κ_III`. |
| 3 | Implementar `T(t)`: isotermas (`T_c`, `T_h`) y **adiabáticas** con `T²(t)/κ(t) = cte`. Función `estado_protocolo(t) -> {κ, T, tramo}`. Calcular `τ*` con la fórmula del paper. |
| 4 | Verificar **ciclo cerrado** y continuidad en las fronteras de tramo; comprobar `κ_IV = (κ_III/κ_II)κ_I ≈ 6.2`. Generar tabla `κ(t), T(t)` para control. |
| 5 | Documentar el protocolo (reproducir el esquema de la Fig. 1a) y apoyar el informe. |

**Partes:** parámetros del paper · `κ(t)` cuadrática simétrica · `T(t)` isotermas+adiabáticas · `τ*` · verificación de ciclo cerrado.

---

### 👤 Integrante 3 — Orquestador de simulación + compilación · C++

**Objetivo:** unir integrador (1) + protocolo (2), correr muchas realizaciones y escribir los CSV.

| Sesión | Tarea |
|--------|-------|
| 1 | Acuerdo de interfaces. Crear `CMakeLists.txt` y `src/main.cpp` que compile con *stubs* de 1 y 2 (proyecto compila desde el día 1). |
| 2 | Bucle temporal: en cada `dt`, pedir `{κ, T}` al protocolo (2) y avanzar con el integrador (1); registrar al ritmo `Δt=5 ms`. |
| 3 | Bucle de **N realizaciones** (distinta semilla) para promediar el ruido. Escribir CSV con el formato de §4 (`realizacion,tiempo,tramo,x,v,T,kappa`). |
| 4 | Parámetros por línea de comandos / archivo config (`N`, `τ`, `dt`, `T_c`, `T_h`, `κ_*`). **Barrido de `τ` de 10 a 200 ms** (necesario para las Figs. 2 y 3). Integrar el código real de 1 y 2. |
| 5 | Prueba extremo a extremo: compilar, correr el barrido, generar `datos/*.csv`. Coordinar la integración del grupo. |

**Partes:** CMake · `main.cpp` · bucle temporal · bucle de realizaciones · escritura CSV · barrido de `τ` · config.

---

### 👤 Integrante 4 — Termodinámica estocástica · Python

**Objetivo:** calcular `W`, `Q`, `η` y `P` desde los CSV, con las fórmulas del paper (§3).

| Sesión | Tarea |
|--------|-------|
| 1 | Acuerdo de interfaces. Crear `analisis/termodinamica.py`. Lector de CSV (Pandas) usando datos de prueba mientras C++ no esté listo. |
| 2 | **Trabajo** por paso con **Stratonovich** `dW = ½[F_κ(x_t)+F_κ(x_{t+Δt})]·Δκ`, `F_κ=x²/2`; sumar por tramo y promediar sobre realizaciones. |
| 3 | **Calor** `dQ = dU + dE_cin − dW` por tramo (necesita `v` → `dE_cin`); identificar `Q_1..Q_4`. Verificar `⟨W_τ⟩`, `⟨Q_τ⟩` vs τ ajustan a `A+B/τ` (Fig. 2a). |
| 4 | **Eficiencia** `η_τ^(i)` (ec. 1) y las variantes `η₁, η₂, η₃`; **potencia** `P_τ=−⟨W_τ⟩/τ`. Verificar `η→η_C≈0.43` cuasiestático y `η*≈η_CA≈0.25` a `P_max`. |
| 5 | Exportar tabla-resumen (CSV/JSON) con `⟨W⟩, ⟨Q⟩, η, P` por cada `τ` para las figuras; apoyar el informe. |

**Partes:** lector CSV · trabajo (Stratonovich) · calor (Sekimoto + cinética) · eficiencias `η₁₂₃` · potencia · ajustes `A+B/τ` · tabla-resumen.

---

### 👤 Integrante 5 — Visualización, reproducción de figuras e informe · Python

**Objetivo:** reproducir las figuras del paper, analizar y redactar el informe.

| Sesión | Tarea |
|--------|-------|
| 1 | Acuerdo de interfaces. Crear `analisis/visualizacion.py`, `requirements.txt`, carpeta `figuras/`. Graficar una trayectoria de prueba `x(t)`. |
| 2 | Graficar protocolos `κ(t)` y `T(t)` (reproducir **Fig. 1a**) y la trayectoria de la partícula a lo largo del ciclo. |
| 3 | Reproducir la **Fig. 1c (diagrama de Clapeyron):** `F_κ = ⟨x²⟩/2` vs `κ`; el **área encerrada = trabajo neto**. |
| 4 | Reproducir la **Fig. 2:** `⟨W_τ⟩, ⟨Q_τ⟩, ⟨ΔH_τ⟩` vs `τ`; y `P_τ` + `η_τ` vs `1/τ`, marcando `P_max`, `η_C` y `η_CA`. |
| 5 | Reproducir la **Fig. 3:** PDF de la eficiencia estocástica `ρ_{τ,i}(η)` (buscar bimodalidad y colas `∼ η⁻²`). Redactar el **README/informe final** (física, método, resultados, comparación con el paper, conclusiones). |

**Partes:** entorno Python · protocolos (Fig. 1a) · Clapeyron (Fig. 1c) · energética y potencia/eficiencia (Fig. 2) · PDF de eficiencia (Fig. 3) · informe.

---

## 6. Plan global de las 5 sesiones

| Sesión | Meta del grupo | Resultado al final de la clase |
|--------|----------------|--------------------------------|
| **1** | Acordar interfaces y parámetros del paper; esqueleto del proyecto | Compila/ejecuta con *stubs*; cada quien tiene su archivo; parámetros de §2 fijados |
| **2** | Lógica base de cada segmento | BAOAB da un paso; `κ(t)` lista; bucle temporal corre; trabajo calculado; Fig. 1a graficada |
| **3** | Un ciclo completo real | Ciclo completo en C++ → CSV; calor por tramo; Clapeyron (Fig. 1c); validación equipartición |
| **4** | Barrido de `τ` + resultados | Pipeline C++→CSV→Python; `η` y `P`; Fig. 2 reproducida; `η_C` y `η_CA` verificadas |
| **5** | PDFs, validación final e informe | Fig. 3 (PDF eficiencia); todo validado contra el paper; figuras finales e informe entregable |

---

## 7. Criterios de éxito (validación contra el artículo)

El proyecto está "bien reproducido" si:

1. **Equipartición:** el integrador cumple `⟨x²⟩ = k_B T/κ` en equilibrio (Seg. 1).
2. **Ciclo cerrado** con los `κ_I..κ_IV` exactos y `T²/κ=cte` en adiabáticas (Seg. 2).
3. **Clapeyron (Fig. 1c):** el área del ciclo `F_κ–κ` ≈ trabajo neto (Segs. 4–5).
4. **Energética (Fig. 2a):** `⟨W_τ⟩` y `⟨Q_τ⟩` siguen `A + B/τ`.
5. **Límites teóricos:** `η → η_C ≈ 0.43` cuasiestático; `η* ≈ η_CA ≈ 0.25` a potencia máxima;
   `P_max ≈ 6.34 k_B T_c/s`.
6. **Fluctuaciones (Fig. 3):** PDF de eficiencia **bimodal**, colas `∼ η⁻²`, con eventos "super-Carnot".

---

## 8. Reglas de trabajo en equipo

- **Ramas Git por integrante** (`feat/langevin`, `feat/ciclo`, …) y *pull requests*; nadie empuja directo a `main`.
- **Stubs primero:** cada módulo expone su función vacía desde la Sesión 1, así nadie queda bloqueado.
- **Dependencias:** Seg. 3 necesita las *interfaces* de 1 y 2 (no la implementación); Segs. 4 y 5 necesitan
  el *formato CSV* (no los datos reales) → trabajan con datos de prueba hasta la Sesión 4.
- **Extensiones opcionales** (si sobra tiempo / para nota extra): comparar las 3 definiciones de eficiencia
  y sus fluctuaciones (Métodos del paper); estudiar el régimen de no-equilibrio (ciclos rápidos, `τ=30 ms`).

---

## Referencia

Martínez, I. A., Roldán, É., Dinis, L., Petrov, D., Parrondo, J. M. R. & Rica, R. A.
*Brownian Carnot engine.* **Nature Physics** 12, 67–70 (2016). DOI: 10.1038/nphys3518.
