// =============================================================================
//  langevin.hpp  --  SEGMENTO 1  (Integrador de Langevin con inercia, BAOAB)
//  Dueño: Integrante 1
// -----------------------------------------------------------------------------
//  Corazon numerico del motor: avanza la particula (posicion x y velocidad v)
//  un paso de tiempo dt, resolviendo la ecuacion de Langevin CON inercia:
//
//      m (d2x/dt2) = -kappa(t) x - gamma (dx/dt) + sqrt(2 gamma kB T(t)) xi(t)
//
//  usando el esquema BAOAB (estable y correcto en muestreo de Boltzmann).
//
//  UNIDADES DEL PROYECTO (acordadas en Sesion 1):
//    longitud  : micrometros  (um)
//    fuerza    : piconewtons   (pN)        => rigidez kappa en pN/um
//    tiempo    : milisegundos  (ms)
//    temperatura: kelvin        (K)
//    energia   : pN*um  (coherente con lo anterior)
//  OJO: kB debe expresarse en estas unidades (ver langevin.cpp).
// =============================================================================
#ifndef LANGEVIN_HPP
#define LANGEVIN_HPP

#include <cstdint>
#include <random>

// -----------------------------------------------------------------------------
//  Estado de la particula en un instante dado.
//  Esta struct es la interfaz COMPARTIDA por los Segmentos 1, 2 y 3.
//  NO cambiar sus campos sin avisar al grupo (romperia el contrato).
// -----------------------------------------------------------------------------
struct Estado {
    double x = 0.0;  // posicion [um]
    double v = 0.0;  // velocidad [um/ms]
    double t = 0.0;  // tiempo [ms]
};

// -----------------------------------------------------------------------------
//  Parametros fisicos de la particula y del bano. Los fija el Integrante 1
//  (Sesion 4) a partir de una esfera de poliestireno de 1 um en agua.
// -----------------------------------------------------------------------------
struct ParametrosFisicos {
    double masa   = 0.0;  // m    [pN*ms^2/um]   (masa efectiva de la esfera)
    double gamma  = 0.0;  // gamma[pN*ms/um]     (friccion de Stokes 6*pi*eta*r)
    double kB     = 0.0;  // constante de Boltzmann en unidades del proyecto
};

// -----------------------------------------------------------------------------
//  Integrador de Langevin (BAOAB). Mantiene su propio generador de numeros
//  aleatorios para que cada realizacion sea reproducible con una semilla.
// -----------------------------------------------------------------------------
class IntegradorLangevin {
public:
    IntegradorLangevin(const ParametrosFisicos& fis, std::uint64_t semilla);

    // Avanza el estado un paso dt bajo rigidez kappa y temperatura T (del
    // protocolo del Segmento 2). Devuelve el nuevo estado.
    //   dt    : paso de integracion [ms]  (debe ser << Delta t de muestreo)
    //   kappa : rigidez instantanea [pN/um]
    //   T     : temperatura instantanea del bano [K]
    Estado paso(const Estado& actual, double kappa, double T, double dt);

    // Reinicia la semilla del generador (util para empezar una realizacion).
    void reiniciar_semilla(std::uint64_t semilla);

    const ParametrosFisicos& fisica() const { return fis_; }

private:
    ParametrosFisicos fis_;
    std::mt19937_64 rng_;
    std::normal_distribution<double> gauss_{0.0, 1.0};

    // Devuelve una muestra N(0,1).
    double ruido_normal();
};

#endif  // LANGEVIN_HPP
