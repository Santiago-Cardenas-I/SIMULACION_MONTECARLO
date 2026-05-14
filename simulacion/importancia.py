import numpy as np
import scipy.stats as stats

def importance_sampling_fallos(limite_horas: float = 50.0, n_samples: int = 100000, seed: int = 42):
    """
    Estima la probabilidad de que un pipeline exceda un límite de horas usando Importance Sampling.
    
    Args:
        limite_horas (float): Límite máximo de horas antes de la cancelación.
        n_samples (int): Número de muestras a generar.
        seed (int): Semilla para reproducibilidad.
        
    Returns:
        tuple: (probabilidad_estimada, varianza_del_estimador)
    """
    np.random.seed(seed)
    
    # Target f(x): Lognormal modelando el tiempo del pipeline (cola pesada)
    # Parámetros arbitrarios para la simulación
    s_f = 0.8
    scale_f = np.exp(3.0) 
    
    # Proposal g(x): Exponencial desplazada hacia el límite crítico
    # Concentra las muestras donde x > limite_horas
    loc_g = limite_horas
    scale_g = 10.0
    
    # 1. Muestreamos de la propuesta g(x)
    muestras_g = stats.expon.rvs(loc=loc_g, scale=scale_g, size=n_samples)
    
    # 2. Evaluamos densidades f(x) y g(x)
    pdf_f = stats.lognorm.pdf(muestras_g, s=s_f, scale=scale_f)
    pdf_g = stats.expon.pdf(muestras_g, loc=loc_g, scale=scale_g)
    
    # 3. Calculamos los pesos W = f(x) / g(x)
    pesos = pdf_f / pdf_g
    
    # 4. Función indicadora: 1 si supera el límite, 0 en caso contrario
    indicadora = (muestras_g > limite_horas).astype(float)
    
    # 5. Estimación final
    probabilidad_is = np.mean(indicadora * pesos)
    varianza_is = np.var(indicadora * pesos) / n_samples
    
    return probabilidad_is, varianza_is