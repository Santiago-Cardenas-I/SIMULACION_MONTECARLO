import numpy as np

def p_target(x: np.ndarray, mu1: float=14, sig1: float=2, mu2: float=2, sig2: float=1) -> np.ndarray:
    """
    Densidad bimodal no normalizada p*(x) que imita tráfico diurno y nocturno.
    """
    term1 = np.exp(-((x - mu1)**2) / (2 * sig1**2))
    term2 = 0.5 * np.exp(-((x - mu2)**2) / (2 * sig2**2))
    return term1 + term2

def rejection_sampling_bimodal(n_samples: int = 10000, k: float = 20.0, 
                               limite_inf: float = 0, limite_sup: float = 24, 
                               seed: int = 42):
    """
    Genera muestras bimodales usando Rejection Sampling con envolvente Uniforme.
    
    Args:
        n_samples (int): Muestras deseadas.
        k (float): Constante empírica tal que k*q(x) >= p*(x).
        limite_inf, limite_sup (float): Rango del dominio (ej. 0 a 24 horas).
        seed (int): Semilla para reproducibilidad.
        
    Returns:
        tuple: (array_muestras_aceptadas, tasa_de_aceptacion)
    """
    np.random.seed(seed)
    muestras_aceptadas = []
    intentos_totales = 0
    
    # Muestreo por lotes para maximizar la eficiencia de NumPy
    batch_size = n_samples * 3 
    q_x = 1 / (limite_sup - limite_inf) # Densidad Uniforme
    
    while len(muestras_aceptadas) < n_samples:
        # 1. z ~ q(z) (Distribución Uniforme)
        z = np.random.uniform(limite_inf, limite_sup, batch_size)
        
        # 2. u ~ Uniforme(0, k * q(z))
        u = np.random.uniform(0, k * q_x, batch_size)
        
        # 3. Evaluar condición u <= p*(z)
        p_z = p_target(z)
        aceptados_batch = z[u <= p_z]
        
        # Guardar y contar
        muestras_aceptadas.extend(aceptados_batch)
        intentos_totales += batch_size
        
    muestras_finales = np.array(muestras_aceptadas[:n_samples])
    tasa_aceptacion = n_samples / intentos_totales
    
    return muestras_finales, tasa_aceptacion