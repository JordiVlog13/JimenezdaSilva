import os
import time
import numpy as np
import matplotlib.pyplot as plt
import numba
from numba import prange

# Constantes
N = 200
PI = np.pi
n_ciclos = N/4
lam = 0.5       # 'lambda' es palabra reservada en Python
pasos = 1000

@numba.njit(cache=True, parallel=True, fastmath=True)
def calcular_norma(phi):
    """
    Calcula la norma (L2) del vector de amplitudes phi de manera eficiente, usando paralelización.
    """
    suma = 0.0
    # Paralelizar reducción
    for i in prange(phi.shape[0]):
        re = phi[i].real
        im = phi[i].imag
        suma += re*re + im*im
    return np.sqrt(suma)

@numba.njit(cache=True, parallel=True, fastmath=True)
def generar_arrays():
    """
    Inicializa y devuelve k, s, V, phi, gamma y alpha, usando prange para paralelizar.
    """
    k = 2.0 * PI * n_ciclos / N
    s = 1.0 / (4 * k * k)
    x0 = N / 4
    sigma = N / 16

    V = np.zeros(N+1)
    phi = np.zeros(N+1, dtype=np.complex128)
    norma_loc = 0.0
    # Inicialización paralela con reducción
    for j in prange(N+1):
        if 2*N/5 <= j <= 3*N/5:
            V[j] = lam * k * k
        phi[j] = np.exp(1j * k * j) * np.exp(-((j - x0)**2)/(2*sigma*sigma))
        norma_loc += phi[j].real*phi[j].real + phi[j].imag*phi[j].imag

    # Fronteras y normalización secuencial
    phi[0] = 0
    phi[N] = 0
    norma_loc = np.sqrt(norma_loc)
    for j in range(N+1):
        phi[j] /= norma_loc

    gamma = np.zeros(N, dtype=np.complex128)
    alpha = np.zeros(N, dtype=np.complex128)
    alpha[N-1] = 0
    for j in range(N-1, 0, -1):
        gamma[j] = 1.0/(-2 - V[j] + 2.0j/s + alpha[j])
        alpha[j-1] = -gamma[j]

    return k, s, V, phi, gamma, alpha

@numba.njit(cache=True, parallel=False, fastmath=True)
def calcular_beta(s, gamma, beta, phi, alpha, xi):
    """
    Una iteración de Crank-Nicolson para avanzar phi.
    Este paso es secuencial por dependencias de datos.
    """
    # Barrido hacia atrás
    for j in range(N-1, 0, -1):
        beta[j-1] = gamma[j] * (4.0j * phi[j] / s - beta[j])
    # Barrido hacia adelante
    for j in range(N-1):
        xi[j+1] = alpha[j]*xi[j] + beta[j]
        phi[j+1] = xi[j+1] - phi[j+1]


def main():
    # Medir tiempo de ejecución total
    t_start = time.perf_counter()

    cwd = os.getcwd()
    print(f"Directorio de trabajo actual: {cwd}")

    # Inicializar arrays con numba
    k, s, V, phi, gamma, alpha = generar_arrays()
    beta = np.zeros(N, dtype=np.complex128)
    xi = np.zeros(N+1, dtype=np.complex128)

    norma_instantes = np.zeros(pasos)

    filename = os.path.join(cwd, "schrodinger_data.dat")
    try:
        with open(filename, "w") as datos:
            print(f"Escribiendo datos en: {filename}")
            for l in range(pasos):
                # cálculo de norma en paralelo
                norma_instantes[l] = calcular_norma(phi)
                if l % 3 == 0:
                    for j in range(N+1):
                        prob = phi[j].real*phi[j].real + phi[j].imag*phi[j].imag
                        datos.write(f"{j}, {prob}, {V[j]}, {phi[j].real**2}, {phi[j].imag**2}, {norma_instantes[l]}\n")
                    datos.write("\n")
                calcular_beta(s, gamma, beta, phi, alpha, xi)
        print("Archivo schrodinger_data.dat creado correctamente.")
    except Exception as e:
        print(f"Error al crear o escribir el archivo: {e}")
        return
    
    
    # Medir y mostrar tiempo de ejecución
    t_end = time.perf_counter()
    elapsed = t_end - t_start
    print(f"Tiempo total de ejecución: {elapsed:.4f} segundos")

    # Graficar la norma con ajustes de escala
    fig, ax = plt.subplots()
    ax.plot(np.arange(pasos), norma_instantes)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:.6f}"))
    ymin = norma_instantes.min() * 0.999
    ymax = norma_instantes.max() * 1.001
    ax.set_ylim(ymin, ymax)
    ax.set_xlabel('Paso de tiempo')
    ax.set_ylabel('Norma (L2)')
    ax.set_title('Evolución de la norma de la función de onda')
    ax.grid(True)
    plt.show()


if __name__ == "__main__":
    main()
