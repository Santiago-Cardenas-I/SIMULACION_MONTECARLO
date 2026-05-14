import numpy as np

def simular_latencia_rag(n_samples: int = 100000, 
                         mu_emb: float = 10.0, sigma_emb: float = 2.0, 
                         min_search: float = 5.0, max_search: float = 15.0, 
                         mu_llm: float = 3.0, sigma_llm: float = 0.5, 
                         seed: int = 42) -> np.ndarray:
    """
    Simula la latencia de un sistema RAG mediante Monte Carlo estándar.
    
    Args:
        n_samples (int): Número de consultas a simular.
        mu_emb, sigma_emb (float): Parámetros para la generación de embeddings (Normal).
        min_search, max_search (float): Límites de búsqueda en DB vectorial (Uniforme).
        mu_llm, sigma_llm (float): Parámetros de inferencia del LLM (Lognormal).
        seed (int): Semilla para reproducibilidad estadística.
        
    Returns:
        np.ndarray: Vector de latencias totales simuladas.
    """
    np.random.seed(seed)
    
    # Vectorización: Generación de arrays de tamaño n_samples sin bucles for
    t_embeddings = np.random.normal(loc=mu_emb, scale=sigma_emb, size=n_samples)
    t_search = np.random.uniform(low=min_search, high=max_search, size=n_samples)
    t_llm = np.random.lognormal(mean=mu_llm, sigma=sigma_llm, size=n_samples)
    
    # El tiempo total es la suma de los tiempos independientes
    latencia_total = t_embeddings + t_search + t_llm
    
    return latencia_total