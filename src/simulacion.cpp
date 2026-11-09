// =============================================================================
//  simulacion.cpp  --  SEGMENTO 3  (Orquestador de la simulacion)
//  Dueño: Integrante 3
// -----------------------------------------------------------------------------
//  TODO (Integrante 3):
//    Sesion 2: bucle temporal (pedir {kappa,T} al protocolo y avanzar con el
//              integrador); escribir filas cada delta_muestreo.
//    Sesion 3: bucle de N realizaciones (semilla distinta por realizacion).
//    Sesion 4: barrido de tau (10..200 ms) e integracion con el codigo real.
// =============================================================================
#include "simulacion.hpp"

#include <fstream>
#include <iostream>

bool correr_simulacion(const ConfigSimulacion& cfg) {
    std::ofstream out(cfg.salida_csv);
    if (!out) {
        std::cerr << "ERROR: no se pudo abrir " << cfg.salida_csv << "\n";
        return false;
    }

    // Encabezado del CSV (contrato con Python, NO cambiar sin avisar al grupo).
    out << "realizacion,tiempo,tramo,x,v,T,kappa\n";

    ProtocoloCiclo protocolo(cfg.ciclo);

    // -------------------------------------------------------------------------
    // STUB (Sesion 1): una sola fila de ejemplo para comprobar que el pipeline
    // completo (C++ -> CSV -> Python) funciona de punta a punta. Reemplazar por
    // los bucles reales en las Sesiones 2-3.
    // -------------------------------------------------------------------------
    for (int r = 0; r < cfg.n_realizaciones; ++r) {
        IntegradorLangevin integrador(cfg.fisica, cfg.semilla_base + r);
        Estado estado;  // x=0, v=0, t=0

        // Ejemplo de una unica muestra para que haya datos de prueba:
        PuntoProtocolo p = protocolo.en(estado.t);
        out << r << ',' << estado.t << ',' << static_cast<int>(p.tramo) << ','
            << estado.x << ',' << estado.v << ',' << p.T << ',' << p.kappa
            << '\n';

        (void)integrador;  // (todavia no se usa en el stub)
        break;             // solo una realizacion en el stub
    }

    std::cout << "Simulacion (STUB) escrita en " << cfg.salida_csv << "\n";
    return true;
}
