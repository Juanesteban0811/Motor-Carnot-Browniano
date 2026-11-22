%pip install reservoirpy

#Referencia para la instalación:
#Nathan Trouvain, Luca Pedrelli, Thanh Trung Dinh, Xavier Hinaut. ReservoirPy: an Efficient and User-Friendly Library to Design Echo State Networks. 2020. ⟨hal-02595026⟩ https://hal.inria.fr/hal-02595026
import reservoirpy as rpy
from reservoirpy.nodes import Reservoir
#reservoirpy.hyper para automatizar la busqueda de los mejores parametros
rpy.set_seed(42)
#reservoir = Reservoir()

#PArte 1 : datos simulados
#usando lorenz 96 su ec.diferencial



#solucion por runge kuta 4
import numpy as np
import os

# parte 1: datos simulados con lorenz 96, resuelto con runge kutta 4
#esto para definir
def lorenz96(x, F):
    # dx_j/dt = (x_{j+1} - x_{j-2}) * x_{j-1} - x_j + F
    Q = len(x)
    dx = np.zeros(Q)
    for j in range(Q):
        # indices modulo Q -> frontera periodica
        dx[j] = (x[(j+1) % Q] - x[(j-2) % Q]) * x[(j-1) % Q] - x[j] + F
    return dx

def paso_rk4(x, F, dt):
    k1 = lorenz96(x, F)
    k2 = lorenz96(x + 0.5*dt*k1, F)
    k3 = lorenz96(x + 0.5*dt*k2, F)
    k4 = lorenz96(x + dt*k3, F)
    return x + (dt/6.0)*(k1 + 2*k2 + 2*k3 + k4)

def generar_datos(Q=36, F=8.0, dt=0.01, pasos_guardados=20000, cada=5, transitorio=5000):
  # condicion inicial: todo en F y una perturbacion  en x_0 pARA QUE SEA CAHOTIIC
    x = F*np.ones(Q)
    x[0] += 0.01

    # descartamos el transitorio (los datos del inicio pues que se ven muy afectados por las condiciones iniciales que deberian ser arbitrarias)
    for i in range(transitorio):
        x = paso_rk4(x, F, dt)

    # filas = tiempo, columnas = variables
    U = np.zeros((pasos_guardados, Q))
    fila = 0
    paso = 0
    while fila < pasos_guardados:
        x = paso_rk4(x, F, dt)
        if paso % cada == 0:      # guardamos 1 de cada 'cada' pasosss
            U[fila] = x
            fila += 1
        paso += 1

    return U

U = generar_datos()
print("Forma de los datos (tiempo, variables):", U.shape)
print("Primeras 2 filas:\n", U[:2])
#crear la super carpeta de los datoos
os.makedirs("datos", exist_ok=True)
np.save("datos/lorenz96.npy", U)
print("Guardado en datos/lorenz96.npy")

#parte 2 
#determinando los exponentes de lyapunov


# metodo : dos trayectorias 
def lyapunov_dos_trayectorias(x0, F=8.0, dt=0.01, T=500, d0=1e-8, renorm=10):
    Q = len(x0)
    x = x0.copy()
    np.random.seed(0)
    v = np.random.normal(0, 1, Q)
    y = x + d0 * v / np.linalg.norm(v)      # segunda trayectoria, muy cerquita a la original

    n_pasos = int(T/dt)
    suma = 0.0
    n_renorm = 0
    for paso in range(1, n_pasos+1):
        x = paso_rk4(x, F, dt)
        y = paso_rk4(y, F, dt)
        if paso % renorm == 0:
            d = np.linalg.norm(y - x)
            suma += np.log(d/d0)
            n_renorm += 1
            y = x + (y - x) * d0/d          # devolvemos y a distancia d0
    return suma / (n_renorm*renorm*dt)
Lambda1 = lyapunov_dos_trayectorias(U[0])
print("Lambda (dos trayectorias):", Lambda1)

print("tiempo de lyapunov 1/Lambda:", 1/Lambda1)

#entrenamiento de la red neuronal
from reservoirpy.nodes import Reservoir, Ridge, Input, NVAR
from reservoirpy.observables import nrmse
import matplotlib.pyplot as plt


#cada fila es un instante de tiempo y las columnas son las variables, (20000,36)
U = np.load("datos/lorenz96.npy")   #ahh, los 36 salen de aquí, es pq el sistema lorenz tiene 36 variables

#Hay q elegir cierta cantidad de datos para entrenar la red
n_train = 15000
U_train = U[:n_train]
#otra para los test
U_test = U[n_train:]

media = U_train.mean(axis=0)
std = U_train.std(axis=0)

U_train_norm = (U_train - media) / std   #es para normalizar, pq como F=8, la función tanh se satura
U_test_norm = (U_test - media) / std    #también para normalizar

X_train = U_train_norm[:-1]
X_train = X_train + 0.015 * np.random.default_rng(42).standard_normal(X_train.shape)  #Hay que revisar, no funcionaba bien la predicción sin esta línea, esta agrega ruido
Y_train = U_train_norm[1:]


#Revisar por qué no funciona sin agregar ruido y determinar que tanto cambian los resultados cambiando hiperparámetros


#Parámetros
reservoir = Reservoir(units=1000,#cantidad de neuronas
    lr=1.0,                  #hay que mirar el dt y ajustarlo, pq si es muy grande la red no aprende, y si es muy pequeño tarda mucho en aprender
    sr=0.6,
    input_scaling=0.02,       #toca variar esto pq los datos meten 36 variables simultáneamente y eso puede saturar todo
    input_connectivity=0.1,
    rc_connectivity=0.05,    # Conectividad interna
    seed=42
)
readout = Ridge(ridge=1e-4)  #esto es para los pesos de salida y sirve para que la red no sobreajuste y pueda hacer la predicción


entrada = Input()
cuadraticos = NVAR(delay=1, order=2, strides=1) #térmos cuadráticos
ESN = entrada >> reservoir >> readout & entrada >> cuadraticos >> readout   #Todo esto viene de q Lorenz no es lineal, había que corregir la red, se le tuvo que agregar estos términos cuadráticos

ESN.fit(X_train, Y_train, warmup=300)
ESN.reset()


step_warmup = 50
step_prediccion = 300
prediccion = ESN.run(U_test_norm[:step_warmup])  #iniciamos con 50 pasos para estabilizar la red y luego predecimos 300 pasos

predicciones_norm = np.zeros((step_prediccion, U.shape[1]))
u_actual = prediccion[-1:].ravel()  # última predicción del warmup, a partir de aquí empieza a predecir usando lo que ella misma retorna
predicciones_norm[0] = u_actual

for t in range(1, step_prediccion):
    u_actual = ESN(u_actual)  #usamos ahora la predicción
    predicciones_norm[t] = u_actual

Y_pred = predicciones_norm * std + media  #hay q desnormalizar para comparar con datos reales
Y_real = U_test[step_warmup : step_warmup + step_prediccion]


#La predicción no es válida a partir de cierto punto, ya q el sistema está regido por el Tiempo de Lyapunov
#Para nuestros datos generados, es de aproximadamente 1.67 unidades de tiempo. Por lo que entendemos, la predicción es válida apenas para unos 30 pasos aproximadamente.


tiempo = np.arange(step_prediccion) * 0.05
fig, axes = plt.subplots(2, 1, figsize=(11, 7))

#Predicción para una variable, puede verse de 0 a 35
axes[0].plot(tiempo, Y_real[:, 11], label="Lorenz-96 ($x$)", color="black")
axes[0].plot(tiempo, Y_pred[:, 11], label="Entrenamiento ($x$)", color="red")
axes[0].set_ylabel("Amplitud de la variable$x$")
axes[0].set_title("Predicción en Lorenz-96")
axes[0].legend()

print(f"Error Normalizado: {nrmse(Y_real[:], Y_pred[:]):.4f}")
#Error para todas las variables en cada paso de tiempo
error_abs = np.abs(Y_real - Y_pred).T
im = axes[1].imshow(error_abs, aspect="auto", origin="lower",
                    extent=[tiempo[0], tiempo[-1], 0, U.shape[1]], cmap="magma")
axes[1].set_ylabel("Índice de variable $j$")
axes[1].set_xlabel("Tiempo de simulación")
fig.colorbar(im, ax=axes[1], label="Error absoluto")

plt.tight_layout()
plt.savefig("prediccion_lorenz96.png")
plt.show()
