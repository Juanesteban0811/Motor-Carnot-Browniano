// =============================================================================
//  langevin.cpp  --  SEGMENTO 1  (Integrador de Langevin con inercia, BAOAB)
//  Dueño: Integrante 1
// -----------------------------------------------------------------------------
//  TODO (Integrante 1):
//    Sesion 2: implementar el paso BAOAB completo (subpasos B-A-O-A-B).
//    Sesion 3: validar equiparticion  <x^2> = kB T / kappa,  <v^2> = kB T / m.
//    Sesion 4: fijar masa, gamma (Stokes) y un dt estable; documentarlo.
//
//  Esquema BAOAB para  m dv = (-kappa x - gamma v) dt + sqrt(2 gamma kB T) dW:
//    B (medio) : v += (dt/2) * (-kappa x / m)
//    A (medio) : x += (dt/2) * v
//    O (ruido) : v  = c * v + sqrt((1 - c^2) * kB T / m) * N(0,1),
//                con c = exp(-gamma dt / m)
//    A (medio) : x += (dt/2) * v
//    B (medio) : v += (dt/2) * (-kappa x / m)
// =============================================================================
#include "langevin.hpp"

#include <cmath>

IntegradorLangevin::IntegradorLangevin(const ParametrosFisicos& fis,
                                       std::uint64_t semilla)
    : fis_(fis), rng_(semilla) {}

void IntegradorLangevin::reiniciar_semilla(std::uint64_t semilla) {
    rng_.seed(semilla);
}

double IntegradorLangevin::ruido_normal() {
    return gauss_(rng_);
}

Estado IntegradorLangevin::paso(const Estado& actual, double /*kappa*/,
                                double /*T*/, double dt) {
    // -------------------------------------------------------------------------
    // STUB (Sesion 1): devolver el estado avanzado solo en el tiempo, sin
    // dinamica real, para que el proyecto compile y las demas partes trabajen.
    // Reemplazar por el esquema BAOAB en la Sesion 2.
    // -------------------------------------------------------------------------
    Estado siguiente = actual;
    siguiente.t = actual.t + dt;
    return siguiente;
}
