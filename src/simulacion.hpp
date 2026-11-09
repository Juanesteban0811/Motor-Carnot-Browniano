// =============================================================================
//  simulacion.hpp  --  SEGMENTO 3  (Orquestador de la simulacion)
//  Dueño: Integrante 3
// -----------------------------------------------------------------------------
//  Une el integrador (Segmento 1) y el protocolo (Segmento 2): corre N
//  realizaciones del ciclo completo y escribe las trayectorias a un CSV con el
//  formato acordado (frontera C++ -> Python):
//
//      realizacion,tiempo,tramo,x,v,T,kappa
//
//  El Segmento 4 (Python) leera ese CSV para calcular W, Q, eta y P.
// =============================================================================
#ifndef SIMULACION_HPP
#define SIMULACION_HPP

#include <cstdint>
#include <string>

#include "ciclo.hpp"
#include "langevin.hpp"

// Configuracion de una corrida de simulacion.
struct ConfigSimulacion {
    ParametrosFisicos fisica;        // masa, gamma, kB  (Segmento 1)
    ParametrosCiclo   ciclo;         // T_c, T_h, kappa_*, tau  (Segmento 2)
    double        dt            = 0.05;  // paso de integracion [ms] (<< Delta t)
    double        delta_muestreo = 5.0;  // cada cuanto se guarda una fila [ms]
    int           n_realizaciones = 1000;// cuantas corridas se promedian
    std::uint64_t semilla_base  = 12345; // semilla; cada realizacion usa base+i
    std::string   salida_csv    = "datos/trayectorias.csv";
};

// Ejecuta la simulacion completa y escribe el CSV. Devuelve true si todo ok.
bool correr_simulacion(const ConfigSimulacion& cfg);

#endif  // SIMULACION_HPP
