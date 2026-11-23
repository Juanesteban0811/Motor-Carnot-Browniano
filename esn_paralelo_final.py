"""
ESN PARALELO (reservoirs locales) para prediccion de Lorenz-96
================================================================
Arquitectura estilo Pathak et al. (PRL 120, 024102, 2018): en vez de un unico
reservoir global para las 36 variables, se usan reservoirs locales paralelos que
explotan la localidad espacial de Lorenz-96 (cada variable x_j solo interactua con
x_{j-2}, x_{j-1}, x_{j+1}).

Configuracion ganadora (barrido con validacion robusta: umbral 0.4, 8 ventanas):
    18 grupos de 2 variables, solape 2, 600 neuronas por grupo,
    sr=0.4, input_scaling=0.15, lr=1.0, ridge=1e-6, ruido=0.005
    -> horizonte ~4.16 +/- 0.94 tiempos de Lyapunov

Requiere que existan en memoria (de las celdas previas del notebook):
    U, U_train_norm, U_test_norm, media, std
"""
from reservoirpy.nodes import Reservoir, Ridge
import numpy as np
import matplotlib.pyplot as plt

# ======================================================================
#  Parametros del sistema y de la arquitectura
# ======================================================================
Q            = U.shape[1]        # 36 variables
n_grupos     = 18                # 18 grupos
tam_grupo    = Q // n_grupos     # 2 variables centrales por grupo
solape       = 2                 # vecinas a cada lado (minimo para capturar el acoplamiento)

# hiperparametros ganadores del reservoir
units_g       = 600
lr, sr        = 1.0, 0.4
input_scaling = 0.15
ridge_val     = 1e-6
noise_amp     = 0.005
seed          = 42

# parametros fisicos para la validacion
dt_datos   = 0.05                # Δt efectivo entre filas (dt=0.01 * cada=5)
lambda_max = 1.67                # exponente de Lyapunov maximo (Lorenz-96, F=8, N=36)
t_lyapunov = 1.0 / lambda_max    # ≈ 0.599 u.t.

# ======================================================================
#  Indices de cada grupo (frontera periodica, como Lorenz-96)
# ======================================================================
centros  = [np.arange(g*tam_grupo, (g+1)*tam_grupo) for g in range(n_grupos)]
entradas = [np.arange(g*tam_grupo - solape, (g+1)*tam_grupo + solape) % Q
            for g in range(n_grupos)]

# ======================================================================
#  Entrenamiento: un reservoir + readout [r ; r^2] por grupo
# ======================================================================
rng = np.random.default_rng(seed)
Xtr = U_train_norm[:-1] + noise_amp * rng.standard_normal(U_train_norm[:-1].shape)
Ytr = U_train_norm[1:]

reservoirs, readouts = [], []
for g in range(n_grupos):
    res = Reservoir(units=units_g, lr=lr, sr=sr, input_scaling=input_scaling,
                    input_connectivity=0.1, rc_connectivity=0.05, seed=seed + g)
    rd  = Ridge(ridge=ridge_val)
    R   = res.run(Xtr[:, entradas[g]])              # entrada = centrales + solape
    Rf  = np.concatenate([R, R**2], axis=1)         # [r ; r^2] (cuadrados puros)
    rd.fit(Rf[100:], Ytr[100:, centros[g]])         # objetivo = solo las centrales
    reservoirs.append(res)
    readouts.append(rd)

print(f"Entrenados {n_grupos} reservoirs de {units_g} neuronas "
      f"(entrada {len(entradas[0])}, salida {tam_grupo})")

# ======================================================================
#  Prediccion autonoma conjunta
# ======================================================================
def predecir_paralelo(inicio, step_warmup=150, step_prediccion=300):
    for res in reservoirs:
        res.reset()
    # warmup: cada reservoir se sincroniza con su trozo de datos reales
    estados = [np.asarray(reservoirs[g].run(
                 U_test_norm[inicio - step_warmup : inicio, entradas[g]])[-1]).ravel()
               for g in range(n_grupos)]

    pred = np.zeros((step_prediccion, Q))
    u = np.zeros(Q)
    for g in range(n_grupos):                        # primer paso desde el estado del warmup
        r = estados[g]
        u[centros[g]] = readouts[g].run(np.concatenate([r, r**2]).reshape(1, -1)).ravel()
    pred[0] = u

    for t in range(1, step_prediccion):              # reinyeccion autonoma
        un = np.zeros(Q)
        for g in range(n_grupos):
            r = np.asarray(reservoirs[g].run(u[entradas[g]].reshape(1, -1))[-1]).ravel()
            un[centros[g]] = readouts[g].run(np.concatenate([r, r**2]).reshape(1, -1)).ravel()
        u = un
        pred[t] = u
    return pred

# ======================================================================
#  Validacion robusta: horizonte promediado sobre varias ventanas
# ======================================================================
step_warmup, step_prediccion, umbral = 150, 300, 0.4
n_test = len(U_test_norm)
arranques = np.linspace(step_warmup, n_test - step_prediccion - 1, 8).astype(int)

horizontes = []
for ini in arranques:
    pred = predecir_paralelo(ini, step_warmup, step_prediccion)
    real = U_test_norm[ini : ini + step_prediccion]
    err_t = np.sqrt(((pred - real)**2).mean(axis=1))
    cruce = np.argmax(err_t > umbral) if np.any(err_t > umbral) else step_prediccion
    horizontes.append(cruce)

h = np.array(horizontes) * dt_datos / t_lyapunov
print(f"Horizonte ESN PARALELO: {h.mean():.2f} +/- {h.std():.2f} tiempos de Lyapunov")
print(f"(por ventana: {[f'{x:.2f}' for x in h]})")

# ======================================================================
#  Figura: prediccion de una ventana representativa
# ======================================================================
inicio = arranques[0]
pred_norm = predecir_paralelo(inicio, step_warmup, step_prediccion)
Y_pred = pred_norm * std + media
Y_real = U_test_norm[inicio : inicio + step_prediccion] * std + media

err_t = np.sqrt(((pred_norm - U_test_norm[inicio:inicio+step_prediccion])**2).mean(axis=1))
cruce = np.argmax(err_t > umbral) if np.any(err_t > umbral) else step_prediccion
t_valido_ut   = cruce * dt_datos
t_valido_lyap = t_valido_ut / t_lyapunov

tiempo = np.arange(step_prediccion) * dt_datos
fig, axes = plt.subplots(2, 1, figsize=(11, 7))

axes[0].plot(tiempo, Y_real[:, 11], color="black", label="Lorenz-96 ($x_{11}$)")
axes[0].plot(tiempo, Y_pred[:, 11], color="red",   label="ESN paralelo")
axes[0].axvline(t_valido_ut, color="blue", ls="--",
                label=f"Valido: {t_valido_lyap:.1f} T_Lyap")
axes[0].set_ylabel("Amplitud")
axes[0].set_title("Prediccion de Lorenz-96 con ESN paralelo (18 grupos, solape 2)")
axes[0].legend()
secax = axes[0].secondary_xaxis('top',
    functions=(lambda t: t / t_lyapunov, lambda l: l * t_lyapunov))
secax.set_xlabel("Tiempo (en tiempos de Lyapunov)")

error_abs = np.abs(Y_real - Y_pred).T
im = axes[1].imshow(error_abs, aspect="auto", origin="lower",
                    extent=[tiempo[0], tiempo[-1], 0, Q], cmap="magma")
axes[1].set_ylabel("Indice de variable $j$")
axes[1].set_xlabel("Tiempo de simulacion")
fig.colorbar(im, ax=axes[1], label="Error absoluto")

plt.tight_layout()
plt.savefig("prediccion_lorenz96_paralelo.png", dpi=120)
plt.show()
