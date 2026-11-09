# =============================================================================
#  CMakeLists.txt  --  SEGMENTO 3  (Sistema de compilacion del nucleo C++)
#  Dueño: Integrante 3
# -----------------------------------------------------------------------------
#  Compilar:
#      cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
#      cmake --build build
#  Ejecutar:
#      ./build/motor_browniano 200 1000 datos/trayectorias.csv
# =============================================================================
cmake_minimum_required(VERSION 3.15)
project(MotorCarnotBrowniano LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Optimizacion por defecto si no se especifica el tipo de build.
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

# Flags de optimizacion (Integrante 1, Sesion 5).
set(CMAKE_CXX_FLAGS_RELEASE "-O3")

add_executable(motor_browniano
    src/main.cpp
    src/simulacion.cpp
    src/langevin.cpp
    src/ciclo.cpp
)

target_include_directories(motor_browniano PRIVATE src)
