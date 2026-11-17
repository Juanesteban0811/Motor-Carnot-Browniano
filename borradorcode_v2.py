#entrenamiento de la red neuronal
from reservoirpy.nodes import Reservoir, Ridge, Input
# CAMBIADO: quitamos NVAR de los imports. Mas abajo explicamos por que ya no se usa.
from reservoirpy.observables import nrmse
import matplotlib.pyplot as plt
import numpy as np   # CAMBIADO: aseguramos el import de numpy en este bloque


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
# CAMBIADO: bajamos el ruido de 0.015 a 0.001. El ruido ayuda a que la red se
# recupere de sus propios errores en la fase autonoma, pero 0.015 era tan alto
# que degradaba la precision (y por eso el horizonte era corto). Con lr mas bajo
# y ridge mas bajo (ver abajo) la red ya es estable, asi que basta un ruido suave.
X_train = X_train + 0.001 * np.random.default_rng(42).standard_normal(X_train.shape)
Y_train = U_train_norm[1:]


#Revisar por qué no funciona sin agregar ruido y determinar que tanto cambian los resultados cambiando hiperparámetros


#Parámetros
reservoir = Reservoir(
    units=3000,              # CAMBIADO: 1000 -> 3000. Para 36 variables caoticas acopladas 1000 neuronas quedan cortas; el paper usa 5000. Mas neuronas = mas capacidad = horizonte mas largo.
    lr=0.3,                  # CAMBIADO: 1.0 -> 0.3. Este es el cambio clave. lr=1.0 borraba todo el estado anterior en cada paso (sin memoria suave). Lorenz-96 es continuo y muestreado fino, necesita que el reservoir "recuerde" su trayectoria -> lr bajo lo sincroniza con el ritmo real del sistema.
    sr=0.6,                  # (igual) radio espectral, buen punto medio memoria/estabilidad.
    input_scaling=0.05,      # CAMBIADO: 0.02 -> 0.05. Un poco mas de peso a la entrada para que el reservoir "sienta" mejor las 36 variables, sin saturar la tanh.
    input_connectivity=0.1,
    rc_connectivity=0.05,    # Conectividad interna
    seed=42
)
readout = Ridge(ridge=1e-6)  # CAMBIADO: 1e-4 -> 1e-6. 1e-4 regularizaba de mas y suavizaba la prediccion (se desviaba antes). Para predecir caos de forma autonoma conviene un ridge mas bajo, que ajusta mas fino.


# CAMBIADO: eliminamos el NVAR sobre la ENTRADA. El paper (nota [16]) usa r^2 =
# el cuadrado del ESTADO DEL RESERVOIR (las 3000 neuronas), NO el cuadrado de las
# 36 variables de entrada. Eran cosas distintas y por eso el readout no recibia la
# caracteristica correcta. Abajo construimos [r ; r^2] a mano y se lo pasamos al Ridge.
# Esta es, probablemente, la correccion que mas sube el horizonte de prediccion.

# --- Entrenamiento: recolectar estados del reservoir y armar [r ; r^2] ---
reservoir.reset()
# estados del reservoir para toda la secuencia de entrenamiento:
R = reservoir.run(X_train)                       # CAMBIADO: corremos el reservoir para obtener r(t)
R_feat = np.concatenate([R, R**2], axis=1)       # CAMBIADO: caracteristica [r ; r^2] (fiel al paper)
# descartamos el warmup (transitorio) tanto en features como en objetivo:
warmup = 300
readout.fit(R_feat[warmup:], Y_train[warmup:])   # CAMBIADO: entrenamos SOLO la capa de salida con la regresion ridge


# --- Sincronizacion antes de predecir (warmup con datos reales) ---
reservoir.reset()
step_warmup = 150          # CAMBIADO: 50 -> 150. Con 50 pasos el reservoir no alcanzaba el estado correcto antes de soltarlo solo; 150 lo sincroniza mejor (coherente con el warmup=300 del entrenamiento).
step_prediccion = 300

R_wu = reservoir.run(U_test_norm[:step_warmup])  # CAMBIADO: cargamos el estado del reservoir con datos reales
r_actual = R_wu[-1]                              # ultimo estado del reservoir tras el warmup

# primera prediccion a partir del estado sincronizado:
feat = np.concatenate([r_actual, r_actual**2])   # CAMBIADO: [r ; r^2] para el readout
u_actual = readout.run(feat.reshape(1, -1)).ravel()

predicciones_norm = np.zeros((step_prediccion, U.shape[1]))
predicciones_norm[0] = u_actual

for t in range(1, step_prediccion):
    # CAMBIADO: bucle autonomo explicito r -> [r;r^2] -> u -> r ...
    r_actual = reservoir.run(u_actual.reshape(1, -1))[-1]   # avanzamos el reservoir con la propia prediccion
    feat = np.concatenate([r_actual, r_actual**2])          # [r ; r^2]
    u_actual = readout.run(feat.reshape(1, -1)).ravel()     # nueva prediccion
    predicciones_norm[t] = u_actual

Y_pred = predicciones_norm * std + media  #hay q desnormalizar para comparar con datos reales
Y_real = U_test[step_warmup : step_warmup + step_prediccion]


#La predicción no es válida a partir de cierto punto, ya q el sistema está regido por el Tiempo de Lyapunov
# CAMBIADO (aclaracion de numeros): para Lorenz-96 con F=8, Q=36 el EXPONENTE es
# Lambda ~ 1.67, por lo tanto el TIEMPO de Lyapunov es t_L = 1/Lambda ~ 0.6 s.
# (Antes estaba invertido: 1.67 NO es el tiempo de Lyapunov, es el exponente.)

# CAMBIADO: definimos Lambda para medir el horizonte en tiempos de Lyapunov.
Lambda = 1.67           # usar aqui el valor que imprima tu calculo de lyapunov_dos_trayectorias
dt_datos = 0.05         # paso temporal de los datos guardados (dt=0.01 * cada=5)
tiempo = np.arange(step_prediccion) * dt_datos
tiempo_lyap = Lambda * tiempo   # CAMBIADO: eje en unidades de Lyapunov (Lambda * t)

# CAMBIADO: calculamos el RMSE normalizado paso a paso y el "tiempo de prediccion valido"
err = np.sqrt(np.mean(((Y_pred - Y_real) / std)**2, axis=1))   # RMSE normalizado por variable
umbral = 0.4
cruces = np.where(err > umbral)[0]
if len(cruces) > 0:
    t_valido_lyap = tiempo_lyap[cruces[0]]
    print(f"Tiempo de prediccion valido: {t_valido_lyap:.2f} tiempos de Lyapunov "
          f"({tiempo[cruces[0]]:.2f} s) con umbral RMSE={umbral}")
else:
    print("La prediccion no supero el umbral en toda la ventana (muy buena!).")


# CAMBIADO: ahora son 3 paneles (agregamos el RMSE vs tiempo de Lyapunov).
fig, axes = plt.subplots(3, 1, figsize=(11, 10))

#Predicción para una variable, puede verse de 0 a 35
axes[0].plot(tiempo_lyap, Y_real[:, 11], label="Lorenz-96 ($x$)", color="black")   # CAMBIADO: eje en Lambda*t
axes[0].plot(tiempo_lyap, Y_pred[:, 11], label="Predicción ($x$)", color="red")
axes[0].set_ylabel("Amplitud de la variable $x$")
axes[0].set_xlabel("Tiempo de Lyapunov ($\\Lambda t$)")   # CAMBIADO
axes[0].set_title("Predicción en Lorenz-96 (variable j=11)")
axes[0].legend()

print(f"Error Normalizado (nrmse global): {nrmse(Y_real, Y_pred):.4f}")
#Error para todas las variables en cada paso de tiempo
error_abs = np.abs(Y_real - Y_pred).T
im = axes[1].imshow(error_abs, aspect="auto", origin="lower",
                    extent=[tiempo_lyap[0], tiempo_lyap[-1], 0, U.shape[1]], cmap="magma")  # CAMBIADO: eje en Lambda*t
axes[1].set_ylabel("Índice de variable $j$")
axes[1].set_xlabel("Tiempo de Lyapunov ($\\Lambda t$)")   # CAMBIADO
fig.colorbar(im, ax=axes[1], label="Error absoluto")

# CAMBIADO: panel nuevo -> RMSE(t) vs tiempo de Lyapunov, con el umbral marcado.
axes[2].plot(tiempo_lyap, err, color="blue")
axes[2].axhline(umbral, color="gray", ls="--", label=f"umbral={umbral}")
axes[2].set_ylabel("RMSE normalizado")
axes[2].set_xlabel("Tiempo de Lyapunov ($\\Lambda t$)")
axes[2].set_title("Error vs tiempo de Lyapunov (aqui se lee el horizonte de prediccion)")
axes[2].legend()

plt.tight_layout()
plt.savefig("prediccion_lorenz96.png")
plt.show()
