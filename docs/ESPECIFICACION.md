# Especificación Técnica — Simulación de un Motor de Carnot Browniano

**Proyecto:** Sistemas termodinámicos fuera del equilibrio: Simulación de un motor de Carnot browniano
**Fase:** 1 — Especificación (sin implementación)
**Rol:** Arquitectura de software científico
**Estado:** 🟡 BORRADOR PARA REVISIÓN Y APROBACIÓN

> ⚠️ Este documento contiene **puntos de decisión abiertos** marcados con `🔴 DECISIÓN`. No deben implementarse hasta acordarlos. Al final hay un resumen consolidado de las preguntas.

---

## 1. Visión general

El proyecto simula una **máquina térmica a escala molecular**: una única partícula browniana atrapada en un potencial armónico cuya rigidez varía en el tiempo, inmersa en un baño térmico cuya temperatura también cambia según un protocolo que reproduce un **ciclo de Carnot** (dos "isotermas" y dos "adiabáticas" en el sentido browniano).

A diferencia de un motor macroscópico, aquí el trabajo y el calor son **variables estocásticas**: cada realización del motor produce un valor distinto. Las magnitudes termodinámicas (trabajo, calor, eficiencia, potencia) solo tienen sentido como **promedios sobre un ensamble** de trayectorias, con sus fluctuaciones. Este es el corazón conceptual del proyecto y condiciona toda la arquitectura.

El marco teórico es la **termodinámica estocástica** (Sekimoto, Seifert), donde el calor y el trabajo se definen a nivel de trayectoria individual.

---

## 2. Requisitos funcionales

### RF — Núcleo de simulación
- **RF1.** Integrar numéricamente la ecuación de Langevin para una partícula browniana 1D en un potencial armónico.
- **RF2.** Soportar un potencial armónico `U(x,t) = ½·k(t)·x²` con rigidez `k(t)` dependiente del tiempo (controlada por un protocolo).
- **RF3.** Soportar temperatura del baño `T(t)` dependiente del tiempo (protocolo de temperatura).
- **RF4.** Implementar el **protocolo del ciclo de Carnot browniano**: 2 ramas isotérmicas + 2 ramas adiabáticas, parametrizable (duraciones, valores de `k` y `T` en los puntos de conmutación).
- **RF5.** Simular **N trayectorias estocásticas independientes** del mismo protocolo (ensamble), con semillas reproducibles.

### RF — Termodinámica estocástica
- **RF6.** Calcular, a lo largo de cada trayectoria, el **trabajo** `W` según la definición de Sekimoto (ver §4).
- **RF7.** Calcular el **calor** `Q` intercambiado con el baño.
- **RF8.** Calcular la **energía interna** `U` (o su cambio `ΔU`) y verificar la **primera ley** `ΔU = Q − W` (con el convenio de signos a fijar).
- **RF9.** Promediar sobre el ensamble para obtener `⟨W⟩`, `⟨Q⟩`, y sus desviaciones estándar / histogramas.
- **RF10.** Calcular **eficiencia** `η = −⟨W_net⟩ / ⟨Q_in⟩` (convenio a fijar) y **potencia** `P = −⟨W_net⟩ / τ_ciclo`.
- **RF11.** Calcular la **eficiencia de Carnot** `η_C = 1 − T_fría/T_caliente` y comparar.

### RF — Análisis y visualización
- **RF12.** Generar el **diagrama termodinámico** del ciclo (análogo a P–V; típicamente `⟨k⟩` vs `⟨x²⟩` o fuerza generalizada vs desplazamiento generalizado — ver §4).
- **RF13.** Visualizar trayectorias individuales `x(t)`, protocolos `k(t)`, `T(t)`.
- **RF14.** Histogramas / distribuciones de `W`, `Q` sobre el ensamble (fluctuaciones).
- **RF15.** Curvas de eficiencia y potencia vs parámetros del protocolo (p. ej. vs duración del ciclo → trade-off potencia/eficiencia).
- **RF16.** Exportar resultados a disco (CSV/Parquet vía Pandas; figuras PNG/SVG).

### RF — Calidad y reproducibilidad
- **RF17.** Reproducibilidad total vía semillas RNG explícitas.
- **RF18.** Configuración de cada experimento vía archivo (p. ej. YAML/JSON) o dataclass, no valores "hardcoded".
- **RF19.** Pruebas de validación automatizadas (ver §6) — *solo si el usuario las aprueba; no se añaden por defecto.*

### RNF — No funcionales (propuestos)
- **RNF1.** Rendimiento: ensambles de `N ~ 10³–10⁵` trayectorias en tiempo razonable (vectorización con NumPy sobre el ensamble).
- **RNF2.** Precisión numérica controlable vía `dt`; convergencia verificable.
- **RNF3.** Código documentado (docstrings, unidades explícitas de cada magnitud).

---

## 3. Modelo físico y matemático

### 3.1 Ecuación de Langevin

Dos regímenes posibles. **🔴 DECISIÓN D1 — régimen dinámico:**

**(a) Langevin subamortiguada (con inercia):**
```
m ẍ = −γ ẋ − ∂U/∂x + √(2 γ k_B T(t)) · ξ(t)
```

**(b) Langevin sobreamortiguada (overdamped, sin inercia):**
```
γ ẋ = −∂U/∂x + √(2 γ k_B T(t)) · ξ(t)
```
donde `ξ(t)` es ruido blanco gaussiano: `⟨ξ(t)⟩ = 0`, `⟨ξ(t)ξ(t')⟩ = δ(t−t')`.

> El **régimen sobreamortiguado** es el estándar en la literatura de motores brownianos (es el límite de las trampas ópticas / coloides) y simplifica tanto la integración como las definiciones de calor/trabajo. El subamortiguado es más rico pero complica la adiabática (ver D4). **Recomendación del arquitecto:** empezar con (b) sobreamortiguado.

### 3.2 Potencial armónico
```
U(x,t) = ½ · k(t) · x²
∂U/∂x = k(t) · x
```
La rigidez `k(t) > 0` es el **parámetro de control** (análogo al volumen/presión en el motor clásico).

### 3.3 Variable termodinámica conjugada
Para un potencial armónico en equilibrio instantáneo, `⟨x²⟩_eq = k_B T / k`. El par de variables de estado "tipo P–V" más usado es:
- "Volumen" generalizado ↔ `⟨x²⟩` (o su raíz, la anchura de la trampa).
- "Presión"/fuerza generalizada ↔ `k` (o `−∂F/∂(⟨x²⟩)`).

**🔴 DECISIÓN D2 — representación del diagrama del ciclo** (qué ejes usar para el "P–V browniano").

### 3.4 Ciclo de Carnot browniano

Cuatro ramas (análogas al ciclo clásico):
1. **Isoterma caliente** `T = T_h`: se varía `k` de `k_A → k_B`.
2. **Adiabática de expansión**: `T` baja de `T_h → T_c` mientras `k` varía.
3. **Isoterma fría** `T = T_c`: se varía `k` de `k_C → k_D`.
4. **Adiabática de compresión**: `T` sube de `T_c → T_h`.

**🔴 DECISIÓN D3 — definición de "adiabática browniana".** En termodinámica estocástica una adiabática no es simplemente "baño apagado". Opciones usadas en la literatura:
- (i) **Adiabática isentrópica**: imponer `T(t)/√k(t) = const` (o `k/T²`, según convención) de modo que la entropía del estado gaussiano se conserve; el baño sigue presente pero `T` y `k` cambian acopladas.
- (ii) **Adiabática sin contacto térmico**: desconectar el término de ruido (`T→0` o baño ausente) — más delicado numéricamente y físicamente discutible en sobreamortiguado.
- (iii) Protocolos "instantáneos" (quenches) en los extremos.

Esta es **la decisión física más importante** del proyecto y define qué significa "Carnot" aquí.

**🔴 DECISIÓN D4 — compatibilidad régimen/adiabática.** En el límite sobreamortiguado estricto la energía cinética se promedia y la noción de adiabática cambia; conviene decidir D1 y D3 de forma conjunta.

**🔴 DECISIÓN D5 — forma funcional de los protocolos** `k(t)` y `T(t)` dentro de cada rama: lineal, exponencial, u óptima (minimizando trabajo irreversible). Para un primer motor funcional, lineal o la forma isentrópica analítica suele bastar.

### 3.5 Termodinámica estocástica (definiciones de Sekimoto)

Para una trayectoria dada, con `U(x,λ)` y parámetro de control `λ(t)` (aquí `λ = k`):

- **Trabajo** (el control hace trabajo sobre el sistema al cambiar `λ`):
  ```
  dW = (∂U/∂λ) · dλ = ½ x² dk
  ```
- **Calor** (intercambio con el baño por el movimiento a `λ` fijo):
  ```
  dQ = (∂U/∂x) · dx   (interpretación de Stratonovich)
  ```
- **Primera ley** (por trayectoria): `dU = dW + dQ`.

**🔴 DECISIÓN D6 — convenio de signos** (qué es `W>0`: trabajo *sobre* el sistema o *extraído* por el motor) y qué cuenta como `Q_in`. Debe fijarse para que `η` y la comparación con `η_C` sean coherentes.

**🔴 DECISIÓN D7 — interpretación estocástica** (Itô vs Stratonovich). El calor de Sekimoto requiere Stratonovich; esto afecta el esquema numérico y el post-procesado. Hay que ser consistentes.

### 3.6 Unidades
**🔴 DECISIÓN D8 — sistema de unidades.** Recomendación: **unidades reducidas / adimensionales** (`k_B = 1`, `γ = 1`, escalas elegidas), lo que simplifica y es estándar. Alternativa: unidades físicas de un experimento real (trampa óptica, coloide en agua). Definir esto fija todos los números del proyecto.

---

## 4. Métodos numéricos

### 4.1 Integradores de SDE

**🔴 DECISIÓN D9 — esquema de integración:**
- **Euler–Maruyama** (orden fuerte 0.5): simple, robusto, estándar para empezar. Para potencial lineal (armónico) es exacto en la deriva por paso si se usa el esquema adecuado.
- **Esquema de Milstein**: orden 1.0 (con ruido aditivo como aquí, Milstein ≡ Euler–Maruyama, así que no aporta).
- **Esquema exacto para Ornstein–Uhlenbeck**: como el sistema es lineal (OU con coeficientes variables), existe una **actualización exacta por paso** (propagador gaussiano) que es incondicionalmente estable y muy precisa. **Recomendación fuerte del arquitecto** para el régimen sobreamortiguado.
- Para el subamortiguado: esquemas tipo **BAOAB** (Langevin splitting), muy usados y precisos.

El ruido aditivo hace que Itô y Stratonovich coincidan **en la integración de la SDE**; la distinción importa en el **cálculo del calor** (§3.5).

### 4.2 Cálculo de observables termodinámicos
- `W = Σ ½ x_i² (k_{i+1} − k_i)` (trabajo como integral sobre el protocolo).
- `Q = Σ (∂U/∂x)·Δx` con **convención de punto medio (Stratonovich)**: usar `x̄ = (x_i + x_{i+1})/2`.
- `ΔU = U_final − U_inicial`, verificando `ΔU ≈ W + Q` por trayectoria (test de consistencia).

### 4.3 Promedios de ensamble
- Vectorización: simular las `N` trayectorias como un array `(N,)` que avanza en paralelo paso a paso → aprovecha NumPy.
- Estimadores: media, varianza, error estándar `σ/√N`, histogramas.

### 4.4 Parámetros numéricos
**🔴 DECISIÓN D10 — resolución temporal.** Elegir `dt`, número de pasos por rama y `N` (tamaño de ensamble) tras un **estudio de convergencia**. Propuesta inicial: `N = 10⁴`, `dt` tal que `dt·k/γ ≪ 1`.

---

## 5. Arquitectura del software

Diseño modular, cada capa testeable de forma aislada. Propuesta de estructura:

```
Motor-Carnot-Browniano/
├── src/brownian_carnot/
│   ├── __init__.py
│   ├── config.py          # dataclasses de configuración (protocolo, numérico, ensamble)
│   ├── potential.py       # U(x,t), fuerza, ∂U/∂k  (potencial armónico)
│   ├── protocol.py        # k(t), T(t): ramas isoterma/adiabática, ciclo de Carnot
│   ├── langevin.py        # integradores SDE (Euler–Maruyama / OU exacto / BAOAB)
│   ├── simulator.py       # orquesta ensamble de N trayectorias; guarda x(t)
│   ├── thermodynamics.py  # W, Q, ΔU por trayectoria; 1ª ley; eficiencia, potencia
│   ├── ensemble.py        # promedios, varianzas, histogramas (Pandas)
│   └── io.py              # carga config, exporta resultados (CSV/Parquet)
├── analysis/
│   └── plots.py           # Matplotlib: trayectorias, diagrama del ciclo, histogramas
├── experiments/
│   └── run_carnot.py      # script principal de un experimento reproducible
├── configs/
│   └── baseline.yaml      # un protocolo de referencia
├── tests/                 # (opcional, si se aprueba) validaciones de §6
├── docs/
│   └── ESPECIFICACION.md  # este documento
└── README.md
```

**Principios de diseño:**
- **Separación física/numérica/análisis.** El potencial y el protocolo no saben nada del integrador; el integrador no sabe nada de termodinámica; la termodinámica opera sobre trayectorias ya simuladas.
- **Inversión de dependencias:** el integrador recibe `potential` y `protocol` como objetos (permite cambiar armónico por otro potencial sin tocar el solver).
- **Datos como arrays explícitos con unidades documentadas.**
- **Configuración declarativa** (un experimento = un archivo de config + semilla).

**Flujo de datos:**
```
config → protocol(k(t),T(t)) ─┐
                              ├─→ simulator → trayectorias x(t) (array N×pasos)
         potential(U,F) ──────┘                    │
                                                    ▼
                               thermodynamics (W,Q,ΔU por trayectoria)
                                                    │
                                                    ▼
                               ensemble (⟨·⟩, σ, histogramas) → η, P
                                                    │
                                                    ▼
                                        io (export) + plots (figuras)
```

---

## 6. Estrategia de validación

Validación por capas, de lo analíticamente conocido a lo emergente:

1. **V1 — Equilibrio (equipartición).** Con `k` y `T` constantes, verificar `⟨x²⟩ → k_B T / k` y que la distribución estacionaria es gaussiana con esa varianza. Test cuantitativo directo.
2. **V2 — Relajación OU.** Verificar la función de autocorrelación / tiempo de relajación `τ = γ/k` frente a la solución analítica del proceso de Ornstein–Uhlenbeck.
3. **V3 — Primera ley por trayectoria.** `ΔU − (W+Q) ≈ 0` dentro del error numérico, trayectoria a trayectoria.
4. **V4 — Jarzynski / trabajo cuasiestático.** En el límite de protocolo lento, `⟨W⟩ → ΔF` (energía libre). Opcionalmente verificar la **igualdad de Jarzynski** `⟨e^{−βW}⟩ = e^{−βΔF}` — un test fuerte y elegante de termodinámica estocástica.
5. **V5 — Límite cuasiestático del motor.** Cuando `τ_ciclo → ∞`, la eficiencia `η → η_C` y la potencia `P → 0`. Reproducir el **trade-off potencia–eficiencia**.
6. **V6 — Convergencia numérica.** `⟨W⟩`, `⟨Q⟩` estables al reducir `dt`; error de ensamble `∝ 1/√N`.
7. **V7 — Segunda ley.** `η ≤ η_C` siempre (dentro de fluctuaciones); producción de entropía `≥ 0` en promedio.

**🔴 DECISIÓN D11 — alcance de la validación:** cuáles de V1–V7 son obligatorias para "aprobar" el motor. (Recomiendo mínimo V1, V3, V5, V7.)

> Nota: las pruebas automatizadas (tests/) **no se crearán salvo que lo apruebes explícitamente.**

---

## 7. Plan de implementación por etapas

Cada etapa produce algo ejecutable y validable antes de pasar a la siguiente.

| Etapa | Entregable | Validación asociada |
|-------|-----------|---------------------|
| **E0** | Esqueleto del proyecto, config, estructura de paquetes | — |
| **E1** | Potencial armónico + integrador SDE (una partícula, `k`,`T` constantes) | V1, V2 |
| **E2** | Ensamble vectorizado de N trayectorias | V6 (convergencia) |
| **E3** | Protocolos `k(t)`, `T(t)`; ramas isoterma/adiabática; ciclo de Carnot | inspección de `k(t),T(t)` |
| **E4** | Termodinámica estocástica: W, Q, ΔU por trayectoria; 1ª ley | V3, V4 |
| **E5** | Eficiencia, potencia, comparación con η_C | V5, V7 |
| **E6** | Análisis estadístico + visualización (diagrama del ciclo, histogramas, curvas η/P) | — |
| **E7** | Barridos de parámetros + documentación de resultados | trade-off P–η |

**Hito de aprobación por etapa:** al final de E1, E4 y E5 conviene una revisión tuya antes de seguir.

---

## 8. Riesgos técnicos

- **R1.** Definición ambigua de "adiabática browniana" (D3) → el "motor de Carnot" podría no alcanzar `η_C` si se modela mal. *Riesgo conceptual principal.*
- **R2.** Inconsistencia Itô/Stratonovich en el cálculo del calor → viola la 1ª ley. Mitigación: convención de punto medio + tests V3.
- **R3.** Error de discretización en el calor (muy sensible a `dt`). Mitigación: integrador OU exacto + estudio de convergencia.
- **R4.** Fluctuaciones grandes a escala molecular → `η` por trayectoria puede ser absurda (incluso >1 o <0); solo el promedio tiene sentido. Comunicarlo claramente en la visualización.
- **R5.** Coste computacional de ensambles grandes. Mitigación: vectorización; no guardar todas las trayectorias completas si no hace falta.

---

## 9. Resumen de decisiones abiertas (a acordar antes de implementar)

| # | Decisión | Recomendación del arquitecto |
|---|----------|------------------------------|
| **D1** | Régimen: subamortiguado vs **sobreamortiguado** | Sobreamortiguado (overdamped) para empezar |
| **D2** | Ejes del diagrama del ciclo ("P–V" browniano) | `k` vs `⟨x²⟩` |
| **D3** | Definición de adiabática browniana | Isentrópica (`T/√k` = const), baño presente |
| **D4** | Compatibilidad régimen ↔ adiabática | Decidir junto con D1/D3 |
| **D5** | Forma de los protocolos `k(t)`, `T(t)` | Lineal / isentrópica analítica |
| **D6** | Convenio de signos de W y Q | Fijar `W_ext = −W_sys`; `Q_in` del baño caliente |
| **D7** | Interpretación estocástica (Itô/Stratonovich) | Stratonovich para el calor (punto medio) |
| **D8** | Sistema de unidades | Reducidas/adimensionales (`k_B=γ=1`) |
| **D9** | Esquema de integración | OU exacto (lineal); Euler–Maruyama como fallback |
| **D10** | `dt`, pasos, N del ensamble | `N=10⁴` inicial + estudio de convergencia |
| **D11** | Validaciones obligatorias | Mínimo V1, V3, V5, V7 |

---

*Fin de la especificación (Fase 1). Pendiente de revisión y aprobación del usuario.*
