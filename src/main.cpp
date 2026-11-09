// =============================================================================
//  main.cpp  --  SEGMENTO 3  (Punto de entrada del ejecutable C++)
//  Dueño: Integrante 3
// -----------------------------------------------------------------------------
//  Lee parametros (por ahora valores por defecto / linea de comandos simple) y
//  lanza la simulacion. En la Sesion 4 se amplia para el barrido de tau y la
//  lectura de configuracion completa.
//
//  Uso (ejemplo):
//      ./motor_browniano [tau_ms] [n_realizaciones] [salida_csv]
// =============================================================================
#include <cstdlib>
#include <iostream>
#include <string>

#include "simulacion.hpp"

int main(int argc, char** argv) {
    ConfigSimulacion cfg;

    // --- Parametros fisicos de arranque (Integrante 1 los afina en Sesion 4) ---
    // Valores PLACEHOLDER en unidades del proyecto (um, pN, ms, K). Ajustar.
    cfg.fisica.masa  = 1.0e-9;  // TODO: masa esfera poliestireno 1 um
    cfg.fisica.gamma = 1.0e-5;  // TODO: friccion de Stokes 6*pi*eta*r
    cfg.fisica.kB    = 1.0;     // TODO: kB en unidades del proyecto

    // --- Lectura simple de argumentos (ampliar en Sesion 4) ---
    if (argc > 1) cfg.ciclo.tau       = std::atof(argv[1]);
    if (argc > 2) cfg.n_realizaciones = std::atoi(argv[2]);
    if (argc > 3) cfg.salida_csv      = argv[3];

    std::cout << "=== Motor de Carnot Browniano (esqueleto) ===\n"
              << "  tau            = " << cfg.ciclo.tau << " ms\n"
              << "  realizaciones  = " << cfg.n_realizaciones << "\n"
              << "  salida         = " << cfg.salida_csv << "\n";

    if (!correr_simulacion(cfg)) {
        std::cerr << "La simulacion fallo.\n";
        return EXIT_FAILURE;
    }
    return EXIT_SUCCESS;
}
