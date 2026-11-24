"""
esn_paralelo.py  --  MODULO 3: Echo State Network paralelo (estilo Pathak et al.)

Reservoirs locales paralelos para predecir KS 1D. Reune todo lo que aprendimos que
funciona:
  - readout [r ; r^2]  (cuadrados PUROS del estado, sin cruzados) -> fiel al articulo
  - un reservoir por grupo de variables espaciales contiguas
  - cada grupo ve sus variables centrales + un SOLAPE de vecinas (frontera periodica)
  - clip de seguridad en la prediccion autonoma para evitar explosiones numericas

Como KS 1D vive en un anillo de Q puntos, el particionado es 1D (arcos del anillo),
igual que en Lorenz-96. Es el caso que el articulo resuelve.

Requiere: reservoirpy  (pip install reservoirpy)
"""
from __future__ import annotations

import numpy as np
from reservoirpy.nodes import Reservoir, Ridge


class ESNParalelo:
    """Conjunto de reservoirs locales que juntos predicen un sistema espacial 1D periodico."""

    def __init__(self, Q, n_grupos=16, solape=4, units_g=600,
                 lr=1.0, sr=0.4, input_scaling=0.1, ridge=1e-5,
                 noise_amp=0.005, seed=42, clip=6.0):
        assert Q % n_grupos == 0, "Q debe ser divisible por n_grupos"
        self.Q = Q
        self.n_grupos = n_grupos
        self.tam_grupo = Q // n_grupos
        self.solape = solape
        self.clip = clip
        self.noise_amp = noise_amp
        self.seed = seed

        # indices de cada grupo (frontera periodica, modulo Q)
        self.centros = [np.arange(g * self.tam_grupo, (g + 1) * self.tam_grupo)
                        for g in range(n_grupos)]
        self.entradas = [np.arange(g * self.tam_grupo - solape,
                                   (g + 1) * self.tam_grupo + solape) % Q
                         for g in range(n_grupos)]

        # un reservoir + un readout por grupo
        self.reservoirs = [Reservoir(units=units_g, lr=lr, sr=sr,
                                     input_scaling=input_scaling,
                                     input_connectivity=0.1, rc_connectivity=0.05,
                                     seed=seed + g)
                           for g in range(n_grupos)]
        self.readouts = [Ridge(ridge=ridge) for _ in range(n_grupos)]

    # ------------------------------------------------------------------ #
    def entrenar(self, U_train_norm, warmup=100):
        """Entrena cada readout sobre [r ; r^2] con objetivo = sus variables centrales."""
        rng = np.random.default_rng(self.seed)
        Xtr = U_train_norm[:-1] + self.noise_amp * rng.standard_normal(U_train_norm[:-1].shape)
        Ytr = U_train_norm[1:]
        for g in range(self.n_grupos):
            R = self.reservoirs[g].run(Xtr[:, self.entradas[g]])
            Rf = np.concatenate([R, R**2], axis=1)              # [r ; r^2]
            self.readouts[g].fit(Rf[warmup:], Ytr[warmup:, self.centros[g]])
        return self

    # ------------------------------------------------------------------ #
    def predecir(self, U_test_norm, inicio, step_warmup=150, step_prediccion=300):
        """Prediccion autonoma conjunta a partir del instante 'inicio' del test."""
        for res in self.reservoirs:
            res.reset()
        # sincronizacion: cada reservoir carga su estado con datos reales
        estados = [np.asarray(self.reservoirs[g].run(
                     U_test_norm[inicio - step_warmup:inicio, self.entradas[g]])[-1]).ravel()
                   for g in range(self.n_grupos)]

        pred = np.zeros((step_prediccion, self.Q))
        u = np.zeros(self.Q)
        for g in range(self.n_grupos):
            r = estados[g]
            u[self.centros[g]] = self.readouts[g].run(
                np.concatenate([r, r**2]).reshape(1, -1)).ravel()
        pred[0] = u

        for t in range(1, step_prediccion):
            un = np.zeros(self.Q)
            for g in range(self.n_grupos):
                r = np.asarray(self.reservoirs[g].run(
                    u[self.entradas[g]].reshape(1, -1))[-1]).ravel()
                un[self.centros[g]] = self.readouts[g].run(
                    np.concatenate([r, r**2]).reshape(1, -1)).ravel()
            un = np.clip(un, -self.clip, self.clip)             # freno de seguridad
            u = un
            pred[t] = u
        return pred
