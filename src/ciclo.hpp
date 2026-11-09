// =============================================================================
//  ciclo.hpp  --  SEGMENTO 2  (Protocolo del ciclo de Carnot)
//  Dueño: Integrante 2
// -----------------------------------------------------------------------------
//  Define como varian en el tiempo la rigidez kappa(t) y la temperatura T(t)
//  a lo largo de los 4 tramos del ciclo de Carnot browniano, reproduciendo
//  EXACTAMENTE el protocolo de Martinez et al. (Nature Physics 2015, Metodos).
//
//  Parametros exactos del paper:
//    T_c = 300 K,  T_h = 525 K
//    kappa_I   = kappa(0)    = 2.0  pN/um
//    kappa_III = kappa(tau/2)= 20.0 pN/um
//    kappa_II  = kappa(tau/4)= 6.5  pN/um
//    kappa_IV  = (kappa_III/kappa_II) * kappa_I = 6.2 pN/um
//    kappa(t) crece cuadraticamente en [0, tau/2] y es simetrico: kappa(t)=kappa(tau-t)
//    tau* = tau - (tau/2) sqrt((kIII/kII - 1)/(kIII/kI - 1)) ~ 0.76 tau
//    Adiabaticas: T(t) varia con T^2(t)/kappa(t) = constante
// =============================================================================
#ifndef CICLO_HPP
#define CICLO_HPP

// Identificador de cada subproceso del ciclo (coincide con el paper, Fig. 1).
enum class Tramo {
    IsotermaFria     = 1,  // (1) compresion isotermica,  T = T_c
    AdiabaticaComp   = 2,  // (2) compresion adiabatica,  T: T_c -> T_h
    IsotermaCaliente = 3,  // (3) expansion isotermica,   T = T_h
    AdiabaticaExp    = 4   // (4) expansion adiabatica,   T: T_h -> T_c
};

// Resultado del protocolo en un instante t. Interfaz Segmento 2 -> Segmento 3.
struct PuntoProtocolo {
    double kappa = 0.0;             // rigidez [pN/um]
    double T     = 0.0;             // temperatura del bano [K]
    Tramo  tramo = Tramo::IsotermaFria;
};

// Parametros del ciclo (valores por defecto = los exactos del paper).
struct ParametrosCiclo {
    double T_c       = 300.0;  // K
    double T_h       = 525.0;  // K
    double kappa_I   = 2.0;    // pN/um
    double kappa_II  = 6.5;    // pN/um
    double kappa_III = 20.0;   // pN/um
    double tau       = 200.0;  // duracion del ciclo [ms]  (barrido 10..200 ms)
    // kappa_IV y tau* se derivan de los anteriores (ver ciclo.cpp).
};

class ProtocoloCiclo {
public:
    explicit ProtocoloCiclo(const ParametrosCiclo& p);

    // Estado del protocolo (kappa, T, tramo) en el instante t dentro de [0, tau].
    PuntoProtocolo en(double t) const;

    // Valores derivados (los calcula el constructor).
    double kappa_IV() const { return kappa_IV_; }  // = (kIII/kII)*kI ~ 6.2
    double tau_estrella() const { return tau_estrella_; }  // ~ 0.76 tau

    const ParametrosCiclo& parametros() const { return p_; }

private:
    ParametrosCiclo p_;
    double kappa_IV_      = 0.0;
    double tau_estrella_  = 0.0;

    // Ayudantes (a implementar por el Integrante 2):
    double kappa_de_t(double t) const;  // perfil cuadratico simetrico
    double T_de_t(double t) const;      // isotermas + adiabaticas (T^2/kappa=cte)
};

#endif  // CICLO_HPP
