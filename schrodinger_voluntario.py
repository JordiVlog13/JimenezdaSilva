import numpy as np
import cmath
import random
import time
from numba import jit, prange

# Parámetros
N = 500
PI = np.pi
n_ciclos = 50
lam = 0.3
pasos = 2500
experimentos = 1000

# Funciones auxiliares
 
def calcular_norma(phi: np.ndarray) -> float:
    return np.sqrt(np.sum(np.abs(phi)**2))

@jit(cache=True, nopython=True)
def generar():
    k = 2.0 * PI * n_ciclos / N
    s = 1.0 / (4 * k * k)
    V = np.zeros(N+1)
    phi = np.zeros(N+1, dtype= np.complex128)
    gamma = np.zeros(N, dtype= np.complex128)
    alpha = np.zeros(N, dtype= np.complex128)
    # inicialización
    norma = 0.0
    for j in range(N+1):
        if 2*N/5 <= j <= 3*N/5:
            V[j] = lam * k * k
        else:
            V[j] = 0.0
        phi[j] = cmath.exp(1j * k * j) * cmath.exp(-8.0 * (4*j - N)**2 / (N*N))
        norma += abs(phi[j])**2
    phi[0] = phi[N] = 0
    phi /= np.sqrt(norma)
    # tridiagonal coefficients
    alpha[N-1] = 0
    for j in range(N-1, 0, -1):
        gamma[j] = 1.0 / (-2 - V[j] + 2.0j/s + alpha[j])
        alpha[j-1] = -gamma[j]
    return k, s, V, phi, gamma, alpha


def calcular_beta(s, gamma, beta, phi, alpha, xi):
    # backward sweep
    for j in range(N-1, 0, -1):
        beta[j-1] = gamma[j] * (4.0j * phi[j] / s - beta[j])
    # forward sweep
    for j in range(N-1):
        xi[j+1] = alpha[j] * xi[j] + beta[j]
        phi[j+1] = xi[j+1] - phi[j+1]

# Observables y errores
 
def calcular_x(phi: np.ndarray) -> float:
    """Posición media x = sum_j j * |phi_j|^2"""
    j_indices = np.arange(N+1)
    return np.sum(j_indices * np.abs(phi)**2)

 
def calcular_p(phi: np.ndarray) -> float:
    """Momento promedio p = Im(sum_j conj(phi_j)*(phi_{j+1}-phi_j))"""
    diff = phi[1:] - phi[:-1]
    return np.imag(np.vdot(phi[:-1], diff))

 
def calcular_T(phi: np.ndarray) -> float:
    """Energía cinética T = sum_j -conj(phi_j)*(phi_{j+2}-2phi_{j+1}+phi_j)"""
    T = 0+0j
    for j in range(N-1):
        T += -np.conj(phi[j]) * (phi[j+2] - 2*phi[j+1] + phi[j])
    return T.real

 
def calcular_Vobs(V: np.ndarray, phi: np.ndarray) -> float:
    """Energía potencial V = sum_j V_j * |phi_j|^2"""
    return np.sum(V * np.abs(phi)**2)

 
def calcular_E(phi: np.ndarray, V: np.ndarray) -> float:
    """Energía total E = T + V"""
    return calcular_T(phi) + calcular_Vobs(V, phi)


def calcular_PD(phi: np.ndarray) -> float:
    """Probabilidad de transmisión"""
    return np.sum(np.abs(phi[4*N//5 : N+1])**2)


def calcular_maximo(PD: np.ndarray) -> int:
    """Índice del máximo global en PD"""
    return int(np.argmax(PD))

# Errores de discretización (Riemann punto medio)
 
def calcular_error_x(phi: np.ndarray) -> float:
    der2 = np.zeros(N-1)
    for j in range(N-1):
        der2[j] = ((j+2)*abs(phi[j+2])**2 - 2*(j+1)*abs(phi[j+1])**2 + j*abs(phi[j])**2)
    return np.max(der2) * N/24

 
def calcular_error_p(phi: np.ndarray) -> float:
    der2 = np.zeros(N-2, dtype=complex)
    for j in range(N-2):
        der2[j] = (np.conj(phi[j+2])*(phi[j+3]-phi[j+2])
                   -2*np.conj(phi[j+1])*(phi[j+2]-phi[j+1])
                   + np.conj(phi[j])*(phi[j+1]-phi[j]))
    return np.max(np.abs(der2)) * N/24

 
def calcular_error_T(phi: np.ndarray) -> float:
    der2 = np.zeros(N-3)
    for j in range(N-3):
        term = (np.conj(phi[j+2])*(phi[j+4]-2*phi[j+3]+phi[j+2])
                -2*np.conj(phi[j+1])*(phi[j+3]-2*phi[j+2]+phi[j+1])
                + np.conj(phi[j])*(phi[j+2]-2*phi[j+1]+phi[j]))
        der2[j] = abs(term)
    return np.max(der2) * N/24

 
def calcular_error_V(phi: np.ndarray, V: np.ndarray) -> float:
    der2 = np.zeros(N-1)
    for j in range(N-1):
        der2[j] = (V[j+2]*abs(phi[j+2])**2 -2*V[j+1]*abs(phi[j+1])**2 + V[j]*abs(phi[j])**2)
    return np.max(der2) * N/24

 
def calcular_error_E(phi: np.ndarray, V: np.ndarray) -> float:
    return calcular_error_T(phi) + calcular_error_V(phi, V)

# Main
def main():
    import os
    cwd = os.getcwd()  # Directorio actual
    # Variables de salida
    random.seed()
    # Variables de salida
    PD = np.zeros(pasos)
    mi = np.zeros(experimentos, dtype=int)
    mT = 0

    # Abrir archivos de salida
    with open(os.path.join(cwd,'datos_PD.dat'),'w') as f_PD, \
         open(os.path.join(cwd,'datos_observables.dat'),'w') as f_obs:

        # Generar coeficientes iniciales
        k, s, V, phi, gamma, alpha = generar()
        # Condiciones de contorno: beta[N-1]=xi[N]=xi[0]=0
        beta = np.zeros(N, dtype=complex)
        xi = np.zeros(N+1, dtype=complex)
        beta[N-1] = 0+0j
        xi[0]     = 0+0j
        xi[N]     = 0+0j

        start = time.time()
        for m in range(experimentos):
            for l in range(pasos):
                # Probabilidad de transmisión
                PD[l] = calcular_PD(phi)
                #f_PD.write(f"{l} {PD[l]:.6e} \n")

                # Observables y errores cada 10 pasos
                if l % 10 == 0:
                    x   = calcular_x(phi)
                    p   = calcular_p(phi)
                    T   = calcular_T(phi)
                    Vobs= calcular_Vobs(V,phi)
                    E   = calcular_E(phi,V)
                    ex  = calcular_error_x(phi)
                    ep  = calcular_error_p(phi)
                    eT  = calcular_error_T(phi)
                    eV  = calcular_error_V(phi,V)
                    eE  = calcular_error_E(phi,V)
                    #f_obs.write(f"{l} {x:.6e} {ex:.6e} {p:.6e} {ep:.6e} {T:.6e} {eT:.6e} {Vobs:.6e} {eV:.6e} {E:.6e} {eE:.6e} \n")

                # Avanzar un paso
                calcular_beta(s, gamma, beta, phi, alpha, xi)

            # Monte Carlo usando máximo local
            jmax = calcular_maximo(PD)
            

            p = random.random()
            if p > PD[jmax]:
                mi[m] = 1
                mT += 1
            else:
                mi[m] = 0
            
        
        prob = 1.0 * mT / experimentos
        merr = np.sqrt(sum((mi[m] - prob)**2 for m in range(experimentos)) / experimentos)

        end = time.time()

    print(f"Máximo local de PD: {jmax}, valor: {PD[jmax]:.6e}")
    print(f"Probabilidad de transmisión: {prob:.4f} ± {merr:.4f}")
    print(f"Tiempo total de simulación: {end - start:.2f} s")
if __name__ == '__main__':
    main()
