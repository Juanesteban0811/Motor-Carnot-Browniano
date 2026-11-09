// =============================================================================
//  ciclo.cpp  --  SEGMENTO 2  (Protocolo del ciclo de Carnot)
//  Dueño: Integrante 2
// -----------------------------------------------------------------------------
//  TODO (Integrante 2):
//    Sesion 2: implementar kappa_de_t() (cuadratico, simetrico kappa(t)=kappa(tau-t);
//              pasa por kappa_I, kappa_II en tau/4, kappa_III en tau/2).
//    Sesion 3: implementar T_de_t() (isotermas T_c/T_h; adiabaticas con
//              T^2(t)/kappa(t)=cte) y la logica de en() que asigna el Tramo.
//    Sesion 3: calcular tau* con la formula del paper.
//    Sesion 4: verificar ciclo cerrado y que kappa_IV ~ 6.2 pN/um.
// =============================================================================
#include "ciclo.hpp"

#include <cmath>

ProtocoloCiclo::ProtocoloCiclo(const ParametrosCiclo& p) : p_(p) {
    // Valor impuesto por la geometria del ciclo (paper, Metodos):
    kappa_IV_ = (p_.kappa_III / p_.kappa_II) * p_.kappa_I;  // ~ 6.2 pN/um

    // Tiempo de cruce de la adiabatica (paper, Metodos):
    //   tau* = tau - (tau/2) sqrt((kIII/kII - 1)/(kIII/kI - 1))
    const double r1 = (p_.kappa_III / p_.kappa_II) - 1.0;
    const double r2 = (p_.kappa_III / p_.kappa_I) - 1.0;
    tau_estrella_ = p_.tau - (p_.tau / 2.0) * std::sqrt(r1 / r2);  // ~ 0.76 tau
}

double ProtocoloCiclo::kappa_de_t(double /*t*/) const {
    // STUB (Sesion 1): valor constante de arranque. Implementar perfil
    // cuadratico simetrico en la Sesion 2.
    return p_.kappa_I;
}

double ProtocoloCiclo::T_de_t(double /*t*/) const {
    // STUB (Sesion 1): temperatura fria constante. Implementar isotermas +
    // adiabaticas (T^2/kappa=cte) en la Sesion 3.
    return p_.T_c;
}

PuntoProtocolo ProtocoloCiclo::en(double t) const {
    // STUB (Sesion 1): devuelve kappa y T de los ayudantes (aun stubs) y marca
    // siempre el tramo 1. Implementar la seleccion real de tramo en la Sesion 3.
    PuntoProtocolo punto;
    punto.kappa = kappa_de_t(t);
    punto.T     = T_de_t(t);
    punto.tramo = Tramo::IsotermaFria;
    return punto;
}
