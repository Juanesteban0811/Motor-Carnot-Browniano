# =====================================================================================
#  KURAMOTO-SIVASHINSKY 1D + ESN PARALELO  (todo en un solo bloque, para Colab)
# -------------------------------------------------------------------------------------
#  Reproduce el metodo de Pathak et al. (PRL 120, 024102, 2018): reservoir computing
#  con Echo State Networks PARALELAS (reservoirs locales) para predecir un sistema
#  caotico espaciotemporal sin conocer sus ecuaciones.
#
#  Version 1D (la del articulo). Para copiar y pegar en Colab y ejecutar de corrido.
#  Flujo: simular KS -> estimar Lyapunov -> normalizar -> entrenar ESN paralelo
#         -> predecir -> graficar (error en tiempos de Lyapunov).
#
#  Requiere:  pip install reservoirpy   (numpy y matplotlib ya vienen en Colab)
# =====================================================================================

# =====================================================================================
#  PARTE 0 — INSTALAR E IMPORTAR
# =====================================================================================
# En Colab, descomenta la linea de instalacion la primera vez:
# !pip install reservoirpy
import numpy as np
import matplotlib.pyplot as plt
from reservoirpy.nodes import Reservoir, Ridge


# =====================================================================================
#  PARTE 1 — SIMULACION DE KURAMOTO-SIVASHINSKY 1D  (metodo espectral ETDRK4)
# -------------------------------------------------------------------------------------
#  Ecuacion:  y_t = -y*y_x - y_xx - y_xxxx   en dominio periodico de longitud L.
#  Caso canonico: L=22, Q=64 -> sistema caotico. Se integra con ETDRK4 (Kassam &
#  Trefethen 2005) porque la ecuacion es "stiff". Todo con numpy (FFT), sin shenfun.
# =====================================================================================
# NOTA: dt=0.25 con Q=64 hacia explotar la integracion (overflow -> nan). Se corrige
# con dt mas pequeno (0.05) y DEALIASING (regla 2/3), estandar en metodos espectrales.
# Se guarda 1 de cada 'cada' pasos para que el dt EFECTIVO de los datos sea 0.25.
def simular_ks(L=22.0, Q=64, dt=0.05, n_pasos=20000, cada=5, transitorio=10000, semilla=1):
    rng = np.random.default_rng(semilla)
    x = L * np.arange(Q) / Q
    k = 2.0 * np.pi * np.fft.fftfreq(Q, d=L / Q)   # numeros de onda
    Lop = k**2 - k**4                              # operador lineal (-y_xx - y_xxxx) en Fourier

    # --- dealiasing (regla de los 2/3): anula las frecuencias altas que desestabilizan ---
    kmax = np.max(np.abs(k))
    dealias = (np.abs(k) < (2.0 / 3.0) * kmax)

    # --- coeficientes ETDRK4 via cuadratura por contorno ---
    E  = np.exp(dt * Lop)
    E2 = np.exp(dt * Lop / 2.0)
    M  = 32
    r  = np.exp(1j * np.pi * (np.arange(1, M + 1) - 0.5) / M)
    LR = dt * Lop[:, None] + r[None, :]
    Qc = dt * np.real(np.mean((np.exp(LR / 2.0) - 1.0) / LR, axis=1))
    f1 = dt * np.real(np.mean((-4.0 - LR + np.exp(LR) * (4.0 - 3.0 * LR + LR**2)) / LR**3, axis=1))
    f2 = dt * np.real(np.mean((2.0 + LR + np.exp(LR) * (-2.0 + LR)) / LR**3, axis=1))
    f3 = dt * np.real(np.mean((-4.0 - 3.0 * LR - LR**2 + np.exp(LR) * (4.0 - LR)) / LR**3, axis=1))

    y0 = 0.01 * rng.standard_normal(Q)             # condicion inicial pequena
    v  = np.fft.fft(y0)
    g  = -0.5j * k                                 # termino no lineal: -0.5*(y^2)_x

    def no_lineal(vv):
        # termino no lineal con dealiasing aplicado a la entrada y a la salida
        y = np.real(np.fft.ifft(vv * dealias))
        return g * np.fft.fft(y**2) * dealias

    def paso(v):
        Nv = no_lineal(v)
        a  = E2 * v + Qc * Nv
        Na = no_lineal(a)
        b  = E2 * v + Qc * Na
        Nb = no_lineal(b)
        c  = E2 * a + Qc * (2.0 * Nb - Nv)
        Nc = no_lineal(c)
        return E * v + Nv * f1 + 2.0 * (Na + Nb) * f2 + Nc * f3

    for _ in range(transitorio):                   # descartar spin-up
        v = paso(v)
    U = np.zeros((n_pasos, Q))
    fila = 0
    while fila < n_pasos:                           # integrar y guardar 1 de cada 'cada'
        for _ in range(cada):
            v = paso(v)
        U[fila] = np.real(np.fft.ifft(v))
        fila += 1
    return U, x


DT = 0.25                                           # dt EFECTIVO de los datos (dt_interno 0.05 * cada 5)
U, x = simular_ks(L=22.0, Q=64, dt=0.05, cada=5, n_pasos=20000)
print("datos:", U.shape, " rango [%.2f, %.2f]" % (U.min(), U.max()))

# visualizar el campo (debe verse como KS: estructuras onduladas que fluyen)
plt.figure(figsize=(11, 4))
plt.imshow(U[:400].T, aspect="auto", origin="lower", cmap="jet")
plt.xlabel("tiempo"); plt.ylabel("espacio (x)"); plt.title("KS 1D simulado"); plt.colorbar()
plt.show()


# =====================================================================================
#  PARTE 2 — EXPONENTE DE LYAPUNOV MAXIMO  (metodo de dos trayectorias)
# -------------------------------------------------------------------------------------
#  Necesario para medir el horizonte en TIEMPOS DE LYAPUNOV. Debe salir ~0.043 (L=22).
# =====================================================================================
# Usa el MISMO integrador estable (dt pequeno + dealiasing) que la simulacion.
def exponente_lyapunov(y0, L=22.0, Q=64, dt=0.05, n_pasos=50000, d0=1e-7, renorm=5):
    k = 2.0 * np.pi * np.fft.fftfreq(Q, d=L / Q)
    Lop = k**2 - k**4
    kmax = np.max(np.abs(k)); dealias = (np.abs(k) < (2.0 / 3.0) * kmax)
    E  = np.exp(dt * Lop); E2 = np.exp(dt * Lop / 2.0)
    M  = 32; r = np.exp(1j * np.pi * (np.arange(1, M + 1) - 0.5) / M)
    LR = dt * Lop[:, None] + r[None, :]
    Qc = dt * np.real(np.mean((np.exp(LR / 2.0) - 1.0) / LR, axis=1))
    f1 = dt * np.real(np.mean((-4.0 - LR + np.exp(LR) * (4.0 - 3.0 * LR + LR**2)) / LR**3, axis=1))
    f2 = dt * np.real(np.mean((2.0 + LR + np.exp(LR) * (-2.0 + LR)) / LR**3, axis=1))
    f3 = dt * np.real(np.mean((-4.0 - 3.0 * LR - LR**2 + np.exp(LR) * (4.0 - LR)) / LR**3, axis=1))
    g = -0.5j * k

    def no_lineal(vv):
        y = np.real(np.fft.ifft(vv * dealias))
        return g * np.fft.fft(y**2) * dealias

    def paso(v):
        Nv = no_lineal(v)
        a = E2*v + Qc*Nv; Na = no_lineal(a)
        b = E2*v + Qc*Na; Nb = no_lineal(b)
        c = E2*a + Qc*(2.0*Nb - Nv); Nc = no_lineal(c)
        return E*v + Nv*f1 + 2.0*(Na+Nb)*f2 + Nc*f3

    v = np.fft.fft(y0)
    pert = np.zeros(Q); pert[0] = d0
    w = np.fft.fft(y0 + pert)
    suma = 0.0; n_renorm = 0
    for i in range(1, n_pasos + 1):
        v = paso(v); w = paso(w)
        if i % renorm == 0:
            yv = np.real(np.fft.ifft(v)); yw = np.real(np.fft.ifft(w))
            d = np.linalg.norm(yw - yv)
            if d > 0:
                suma += np.log(d / d0); n_renorm += 1
                yw = yv + (yw - yv) * (d0 / d); w = np.fft.fft(yw)
    return suma / (n_renorm * renorm * dt)


lambda_max = exponente_lyapunov(U[0], dt=0.05)     # dt interno pequeno para estabilidad
print("Lambda_max = %.4f   (referencia literatura L=22: ~0.043)" % lambda_max)
print("tiempo de Lyapunov = %.2f" % (1.0 / lambda_max))


# =====================================================================================
#  PARTE 3 — NORMALIZAR Y PARTIR EN ENTRENAMIENTO / TEST
# =====================================================================================
N = U.shape[0]
n_train = int(0.75 * N)
U_train, U_test = U[:n_train], U[n_train:]
media, std = U_train.mean(axis=0), U_train.std(axis=0)
U_train_norm = (U_train - media) / std
U_test_norm  = (U_test  - media) / std
print("train:", U_train_norm.shape, " test:", U_test_norm.shape)


# =====================================================================================
#  PARTE 4 — ESN PARALELO  (reservoirs locales, readout [r ; r^2], solape, clip)
# -------------------------------------------------------------------------------------
#  Lo que aprendimos que funciona:
#    - readout [r ; r^2] = cuadrados PUROS del estado (sin cruzados) -> fiel al articulo
#    - un reservoir por grupo de variables contiguas (explota la localidad espacial)
#    - cada grupo ve sus centrales + un SOLAPE de vecinas (frontera periodica)
#    - clip de seguridad en la prediccion autonoma (evita explosiones numericas)
# =====================================================================================
class ESNParalelo:
    def __init__(self, Q, n_grupos=16, solape=4, units_g=600,
                 lr=1.0, sr=0.4, input_scaling=0.1, ridge=1e-5,
                 noise_amp=0.005, seed=42, clip=6.0):
        assert Q % n_grupos == 0, "Q debe ser divisible por n_grupos"
        self.Q = Q; self.n_grupos = n_grupos; self.tam = Q // n_grupos
        self.solape = solape; self.clip = clip
        self.noise_amp = noise_amp; self.seed = seed
        self.centros  = [np.arange(g*self.tam, (g+1)*self.tam) for g in range(n_grupos)]
        self.entradas = [np.arange(g*self.tam - solape, (g+1)*self.tam + solape) % Q
                         for g in range(n_grupos)]
        self.reservoirs = [Reservoir(units=units_g, lr=lr, sr=sr, input_scaling=input_scaling,
                                     input_connectivity=0.1, rc_connectivity=0.05, seed=seed+g)
                           for g in range(n_grupos)]
        self.readouts = [Ridge(ridge=ridge) for _ in range(n_grupos)]

    def entrenar(self, U_train_norm, warmup=100):
        rng = np.random.default_rng(self.seed)
        Xtr = U_train_norm[:-1] + self.noise_amp * rng.standard_normal(U_train_norm[:-1].shape)
        Ytr = U_train_norm[1:]
        for g in range(self.n_grupos):
            R  = self.reservoirs[g].run(Xtr[:, self.entradas[g]])
            Rf = np.concatenate([R, R**2], axis=1)              # [r ; r^2]
            self.readouts[g].fit(Rf[warmup:], Ytr[warmup:, self.centros[g]])
        return self

    def predecir(self, U_test_norm, inicio, step_warmup=150, step_prediccion=300):
        for res in self.reservoirs:
            res.reset()
        estados = [np.asarray(self.reservoirs[g].run(
                     U_test_norm[inicio-step_warmup:inicio, self.entradas[g]])[-1]).ravel()
                   for g in range(self.n_grupos)]
        pred = np.zeros((step_prediccion, self.Q)); u = np.zeros(self.Q)
        for g in range(self.n_grupos):
            r = estados[g]
            u[self.centros[g]] = self.readouts[g].run(np.concatenate([r, r**2]).reshape(1,-1)).ravel()
        pred[0] = u
        for t in range(1, step_prediccion):
            un = np.zeros(self.Q)
            for g in range(self.n_grupos):
                r = np.asarray(self.reservoirs[g].run(u[self.entradas[g]].reshape(1,-1))[-1]).ravel()
                un[self.centros[g]] = self.readouts[g].run(np.concatenate([r, r**2]).reshape(1,-1)).ravel()
            un = np.clip(un, -self.clip, self.clip)             # freno de seguridad
            u = un; pred[t] = u
        return pred


Q = U.shape[1]
esn = ESNParalelo(Q=Q, n_grupos=16, solape=4, units_g=600,
                  lr=1.0, sr=0.4, input_scaling=0.1, ridge=1e-5,
                  noise_amp=0.005, seed=42, clip=6.0)
esn.entrenar(U_train_norm, warmup=100)
print("ESN paralelo entrenado:", esn.n_grupos, "grupos de", esn.tam,
      "variables (solape", esn.solape, ")")


# =====================================================================================
#  PARTE 5 — PREDICCION AUTONOMA
# =====================================================================================
inicio          = 150
step_warmup     = 150
step_prediccion = 300
predicciones_norm = esn.predecir(U_test_norm, inicio, step_warmup, step_prediccion)
print("prediccion:", predicciones_norm.shape)


# =====================================================================================
#  PARTE 6 — GRAFICA: real vs prediccion + error en tiempos de Lyapunov
# =====================================================================================
umbral = 0.4
real_norm = U_test_norm[inicio : inicio + step_prediccion]
Y_pred = predicciones_norm * std + media
Y_real = real_norm * std + media

err = np.sqrt(((predicciones_norm - real_norm)**2).mean(axis=1))
t_lyap = 1.0 / lambda_max
tiempo_lyap = np.arange(step_prediccion) * DT / t_lyap
cruce = np.argmax(err > umbral) if np.any(err > umbral) else step_prediccion
h_lyap = tiempo_lyap[cruce] if cruce < step_prediccion else tiempo_lyap[-1]
print("Prediccion valida hasta %d pasos = %.2f tiempos de Lyapunov" % (cruce, h_lyap))

fig, axes = plt.subplots(3, 1, figsize=(11, 11))
im0 = axes[0].imshow(Y_real.T, aspect="auto", origin="lower", cmap="jet",
                     extent=[0, tiempo_lyap[-1], 0, Y_real.shape[1]])
axes[0].set_title("KS 1D real"); axes[0].set_ylabel("espacio (x)"); fig.colorbar(im0, ax=axes[0])
im1 = axes[1].imshow(Y_pred.T, aspect="auto", origin="lower", cmap="jet",
                     extent=[0, tiempo_lyap[-1], 0, Y_pred.shape[1]])
axes[1].set_title("Prediccion ESN paralelo"); axes[1].set_ylabel("espacio (x)"); fig.colorbar(im1, ax=axes[1])
axes[2].plot(tiempo_lyap, err, color="blue")
axes[2].axhline(umbral, color="gray", ls="--", label="umbral=%.1f" % umbral)
axes[2].axvline(h_lyap, color="red", ls=":", label="valido: %.1f T_Lyap" % h_lyap)
axes[2].set_xlabel("tiempo (en tiempos de Lyapunov, Lambda*t)")
axes[2].set_ylabel("RMSE normalizado"); axes[2].set_title("Error vs tiempo de Lyapunov"); axes[2].legend()
plt.tight_layout(); plt.savefig("prediccion_ks1d.png", dpi=120); plt.show()


# =====================================================================================
#  PARTE 7 — (OPCIONAL) EVALUACION ROBUSTA SOBRE VARIAS VENTANAS
# =====================================================================================
arranques = [150 + k*300 for k in range(6)
             if 150 + k*300 + step_prediccion <= len(U_test_norm)]
horizontes = []
for ini in arranques:
    p = esn.predecir(U_test_norm, ini, step_warmup, step_prediccion)
    r = U_test_norm[ini:ini+step_prediccion]
    e = np.sqrt(((p - r)**2).mean(axis=1))
    c = np.argmax(e > umbral) if np.any(e > umbral) else step_prediccion
    horizontes.append(c * DT / t_lyap)
horizontes = np.array(horizontes)
print("Horizonte: %.2f +/- %.2f tiempos de Lyapunov (sobre %d ventanas)"
      % (horizontes.mean(), horizontes.std(), len(arranques)))
