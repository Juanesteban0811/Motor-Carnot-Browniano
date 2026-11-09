# Repartición del Proyecto — Motor de Carnot Browniano

**Materia:** Herramientas Computacionales
**Proyecto:** Simulación de un motor térmico browniano mediante ecuaciones de Langevin
**Integrantes:** 5
**Modalidad:** 5 sesiones (clases), un segmento por integrante

---

## 1. Resumen del proyecto

Se simula un **motor de Carnot browniano**: una partícula microscópica (movimiento browniano) atrapada
en un potencial armónico (una "pinza óptica" de rigidez `κ`) sometida a un **ciclo de Carnot** —
dos tramos isotérmicos y dos adiabáticos— variando en el tiempo la **temperatura del baño `T(t)`**
y la **rigidez del potencial `κ(t)`**.

La dinámica de la partícula se modela con la **ecuación de Langevin** (fuerza del potencial + fricción
+ ruido térmico), se integra numéricamente, y a partir de las trayectorias se calculan las cantidades
termodinámicas del ciclo: **trabajo, calor, eficiencia y potencia**.

### Ecuación de Langevin (sobreamortiguada)

```
γ (dx/dt) = -κ(t) · x + √(2 γ k_B T(t)) · ξ(t)
```

donde `γ` es el coeficiente de fricción, `k_B` la constante de Boltzmann, y `ξ(t)` ruido blanco
gaussiano (`<ξ(t)>=0`, `<ξ(t)ξ(t')>=δ(t-t')`).

---

## 2. Stack tecnológico y arquitectura

Se trabaja con **C++** (núcleo de cómputo) y **Python** (análisis y gráficas), que son las herramientas del curso.

| Capa | Lenguaje | Responsabilidad |
|------|----------|-----------------|
| Núcleo de simulación | **C++** | Integrar la ecuación de Langevin a alta velocidad, correr el ciclo y muchas realizaciones |
| Análisis y visualización | **Python** (NumPy, Matplotlib, Pandas) | Leer resultados, calcular termodinámica, graficar, informe |

### Frontera entre C++ y Python

El programa en C++ **escribe los resultados a archivos CSV**; Python los **lee** para analizar y graficar.
Esta es la interfaz que une ambos mundos y debe acordarse entre todos en la Sesión 1.

```
┌────────────────────┐     escribe CSV     ┌────────────────────────┐
│  C++ (segmentos     │ ──────────────────► │  Python (segmentos     │
│  1, 2, 3)           │   datos/*.csv       │  4, 5)                 │
└────────────────────┘                     └────────────────────────┘
```

### Estructura de carpetas propuesta

```
Motor-Carnot-Browniano/
├── src/                     # Código C++
│   ├── langevin.hpp/.cpp     # Segmento 1
│   ├── ciclo.hpp/.cpp        # Segmento 2
│   ├── simulacion.hpp/.cpp   # Segmento 3
│   └── main.cpp              # Segmento 3
├── analisis/                # Código Python
│   ├── termodinamica.py      # Segmento 4
│   └── visualizacion.py      # Segmento 5
├── datos/                   # CSV generados por C++ (entrada de Python)
├── figuras/                 # Gráficas generadas por Python (Segmento 5)
├── CMakeLists.txt o Makefile # Compilación C++ (Segmento 3)
├── requirements.txt          # Dependencias Python (Segmento 5)
└── README.md                 # Informe final (Segmento 5)
```

---

## 3. Contrato de interfaces (acordar en Sesión 1, entre todos)

Para que cada quien trabaje en paralelo sin bloquear a los demás, se fijan primero las "fronteras":

### 3.1 Estado de la partícula (C++, compartido por segmentos 1–3)

```cpp
struct Estado {
    double x;      // posición
    double v;      // velocidad (si se usa modelo con inercia; opcional en sobreamortiguado)
    double t;      // tiempo
    double T;      // temperatura actual del baño
    double kappa;  // rigidez actual del potencial
};
```

### 3.2 Formato del archivo CSV (frontera C++ → Python)

Columnas fijas, separadas por coma, con encabezado:

```
realizacion,tiempo,tramo,x,T,kappa
```

- `realizacion`: índice de la corrida (0..N-1) para promediar sobre el ruido
- `tiempo`: tiempo de simulación
- `tramo`: 0=isoterma caliente, 1=adiabática exp., 2=isoterma fría, 3=adiabática comp.
- `x`: posición de la partícula
- `T`: temperatura del baño en ese instante
- `kappa`: rigidez del potencial en ese instante

> **Importante:** mientras cada integrante no tenga lista su parte, usar *stubs* (funciones vacías
> que devuelven datos de prueba) para no bloquearse. La interfaz de arriba es el acuerdo.

---

## 4. Repartición por integrante y por sesión

Cada integrante es **dueño (owner)** de un segmento. Las sesiones avanzan en paralelo donde es posible
y convergen al final. Al inicio de cada sesión: 10 min de sincronización del grupo.

---

### 👤 Integrante 1 — Núcleo físico: Integrador de Langevin (C++)

**Objetivo:** implementar el corazón numérico que avanza la partícula un paso de tiempo.

| Sesión | Tarea |
|--------|-------|
| 1 | Participar en el acuerdo de interfaces (§3). Crear `src/langevin.hpp/.cpp`. Implementar generador de ruido gaussiano (`std::normal_distribution`) con semilla configurable. |
| 2 | Implementar el paso de integración de Euler–Maruyama para la ecuación de Langevin sobreamortiguada: `x_{n+1} = x_n - (κ/γ)·x_n·dt + √(2 k_B T dt/γ)·N(0,1)`. |
| 3 | Validar el integrador: para `κ` y `T` fijos, verificar que la varianza de equilibrio tiende a `<x²> = k_B T / κ` (teorema de equipartición). Documentar la prueba. |
| 4 | Añadir opción de modelo con inercia (Langevin completo / esquema BAOAB) como extra, y permitir `dt` y `γ` como parámetros. |
| 5 | Ayudar a optimizar (flags de compilación `-O2/-O3`), revisión de código con el grupo y apoyo al informe. |

**Partes del segmento:** generador de ruido · función `paso_langevin()` · validación por equipartición · parámetros físicos.

---

### 👤 Integrante 2 — Protocolo del ciclo de Carnot (C++)

**Objetivo:** definir cómo varían `T(t)` y `κ(t)` a lo largo de los 4 tramos del ciclo.

| Sesión | Tarea |
|--------|-------|
| 1 | Acuerdo de interfaces. Crear `src/ciclo.hpp/.cpp`. Definir parámetros del ciclo: `T_caliente`, `T_fria`, `κ_min`, `κ_max`, duración de cada tramo. |
| 2 | Implementar las dos **isotermas**: `T` constante y `κ(t)` variando (comprimir/expandir el potencial). |
| 3 | Implementar las dos **adiabáticas**: variar `T` y `κ` juntas manteniendo la relación adiabática (sin intercambio neto de calor). Función `estado_protocolo(t) -> {T, κ, tramo}`. |
| 4 | Verificar continuidad del ciclo (que el estado al final del tramo 4 empalme con el inicio del tramo 1) y que el ciclo sea cerrado. |
| 5 | Documentar el ciclo (diagrama de los 4 tramos) y apoyar el informe. |

**Partes del segmento:** parámetros del ciclo · protocolo isotermas · protocolo adiabáticas · función `estado_protocolo(t)` · verificación de ciclo cerrado.

---

### 👤 Integrante 3 — Motor de simulación / orquestador + compilación (C++)

**Objetivo:** unir integrador (1) + protocolo (2) en una corrida completa y escribir los CSV.

| Sesión | Tarea |
|--------|-------|
| 1 | Acuerdo de interfaces. Crear `CMakeLists.txt` (o `Makefile`) y `src/main.cpp` que compile con *stubs* de los segmentos 1 y 2. Que el proyecto compile desde el día 1. |
| 2 | Implementar el bucle temporal: en cada paso, pedir `{T, κ}` al protocolo (2) y avanzar la partícula con el integrador (1). |
| 3 | Implementar el bucle de **N realizaciones** (muchas corridas con distinta semilla) para promediar el ruido. Escribir el CSV con el formato de §3.2. |
| 4 | Añadir lectura de parámetros por archivo de config o argumentos de línea de comandos (`N`, `dt`, `T_c`, `T_f`, …). Integración final con el código real de 1 y 2. |
| 5 | Prueba de extremo a extremo: compilar, correr, generar `datos/*.csv`. Coordinar la integración de todo el grupo. |

**Partes del segmento:** sistema de compilación · `main.cpp` · bucle temporal · bucle de realizaciones · escritura de CSV · parámetros/config.

---

### 👤 Integrante 4 — Termodinámica: trabajo, calor, eficiencia (Python)

**Objetivo:** a partir de los CSV, calcular las cantidades termodinámicas del ciclo.

| Sesión | Tarea |
|--------|-------|
| 1 | Acuerdo de interfaces. Crear `analisis/termodinamica.py`. Función que lee el CSV (Pandas) usando datos de prueba mientras C++ no esté listo. |
| 2 | Implementar cálculo del **trabajo** por tramo: `W = ∫ (∂U/∂κ)·(dκ) `, con `U = ½ κ x²`; es decir `W = ∫ ½ x² dκ`, promediado sobre realizaciones. |
| 3 | Implementar cálculo del **calor** `Q` por tramo (balance de energía: `ΔU = Q - W`) e identificar `Q_absorbido` y `Q_cedido`. |
| 4 | Calcular **eficiencia** `η = W_neto / Q_absorbido`, **potencia** `P = W_neto / T_ciclo`, y comparar con el límite de Carnot `η_C = 1 - T_fria/T_caliente`. |
| 5 | Exportar una tabla-resumen de resultados (CSV/JSON) para el informe y apoyar la redacción. |

**Partes del segmento:** lector de CSV · cálculo de trabajo · cálculo de calor · eficiencia y potencia · comparación con Carnot · tabla de resultados.

---

### 👤 Integrante 5 — Visualización, análisis y documentación (Python)

**Objetivo:** gráficas, análisis de resultados e informe/README final.

| Sesión | Tarea |
|--------|-------|
| 1 | Acuerdo de interfaces. Crear `analisis/visualizacion.py`, `requirements.txt` y la estructura de `figuras/`. Graficar una trayectoria de prueba. |
| 2 | Gráfica de **trayectoria** `x(t)` y de los protocolos `T(t)` y `κ(t)` a lo largo del ciclo. |
| 3 | **Diagrama del ciclo** tipo presión–volumen (análogo browniano: `<x²>` vs `κ`, o fuerza vs posición) mostrando el área encerrada = trabajo neto. |
| 4 | Gráficas de **eficiencia y potencia** vs parámetros (p. ej. vs duración del ciclo o vs `T_fria/T_caliente`), usando resultados del Segmento 4. |
| 5 | Redactar el **README/informe final**: introducción física, metodología, resultados, figuras y conclusiones. Compaginar la entrega del grupo. |

**Partes del segmento:** setup de entorno Python · gráfica de trayectoria · gráfica de protocolos · diagrama del ciclo · gráficas de eficiencia/potencia · informe final.

---

## 5. Plan de las 5 sesiones (vista global del grupo)

| Sesión | Meta del grupo | Resultado al final de la clase |
|--------|----------------|--------------------------------|
| **1** | Acordar interfaces (§3) y crear el esqueleto del proyecto | Proyecto compila/ejecuta con *stubs*; cada quien tiene su archivo creado |
| **2** | Lógica base de cada segmento | Langevin integra; isotermas listas; bucle temporal corre; lector y trayectoria de prueba graficados |
| **3** | Funcionalidad principal completa | Ciclo completo corre en C++ y genera CSV; trabajo y calor calculados; diagrama del ciclo |
| **4** | Integración real (sin stubs) + resultados | Pipeline C++→CSV→Python funcionando; eficiencia y potencia calculadas y graficadas |
| **5** | Pulido, validación e informe | Todo integrado, validado (η ≤ η_Carnot), figuras finales e informe entregable |

---

## 6. Reglas de trabajo en equipo (recomendado)

- **Ramas Git por integrante** (`feat/langevin`, `feat/ciclo`, …) y *pull requests* a `main`; nadie empuja directo a `main`.
- **Stubs primero:** cada módulo expone su función con una implementación vacía desde la Sesión 1, así nadie queda bloqueado esperando a otro.
- **Al inicio de cada sesión:** 10 min de sincronización (qué hizo cada uno, qué se necesita hoy).
- **Dependencias clave:** Segmento 3 necesita las *interfaces* de 1 y 2 (no la implementación) → por eso se acuerdan en Sesión 1. Segmentos 4 y 5 necesitan el *formato CSV* (no los datos reales) → trabajan con datos de prueba hasta la Sesión 4.
