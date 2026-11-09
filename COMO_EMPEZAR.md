# Cómo empezar — Esqueleto del proyecto

Guía rápida para arrancar en la **Sesión 1**. El proyecto ya **compila y corre** de
punta a punta con *stubs*; cada integrante solo rellena su parte.

## Estructura

```
src/            C++  (nucleo de simulacion)
  langevin.*      Segmento 1  (Integrante 1)
  ciclo.*         Segmento 2  (Integrante 2)
  simulacion.*    Segmento 3  (Integrante 3)
  main.cpp        Segmento 3  (Integrante 3)
analisis/       Python (analisis y figuras)
  termodinamica.py  Segmento 4  (Integrante 4)
  visualizacion.py  Segmento 5  (Integrante 5)
datos/          CSV generados por C++   (frontera C++ -> Python)
figuras/        figuras generadas por Python
CMakeLists.txt  compilacion C++
requirements.txt dependencias Python
```

Ver el reparto completo y las tareas por sesión en **`REPARTICION_PROYECTO.md`**.

## Compilar y correr el núcleo C++

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build
./build/motor_browniano 200 1000 datos/trayectorias.csv
#                         ^tau  ^N  ^salida
```

Esto genera `datos/trayectorias.csv` con el encabezado acordado:

```
realizacion,tiempo,tramo,x,v,T,kappa
```

## Análisis en Python

```bash
pip install -r requirements.txt
python analisis/visualizacion.py    # grafica una trayectoria de prueba
python analisis/termodinamica.py    # imprime el resumen termodinamico (stub)
```

## Contrato entre segmentos (no romper sin avisar)

- **`struct Estado`** (`langevin.hpp`): `x`, `v`, `t`. Compartida por Seg. 1–3.
- **`PuntoProtocolo`** (`ciclo.hpp`): `kappa`, `T`, `tramo`. Interfaz Seg. 2 → 3.
- **CSV** `realizacion,tiempo,tramo,x,v,T,kappa`. Frontera C++ → Python (Seg. 3 → 4,5).

Cada función `STUB` lleva comentarios `TODO (Sesion N)` indicando qué implementar y cuándo.
