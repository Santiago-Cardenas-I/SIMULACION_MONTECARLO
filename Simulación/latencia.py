import numpy as np

def simular_latencia_rag(n_samples=100000, seed=42):
    """Calcula el tiempo total de respuesta mediante Monte Carlo estándar."""
    np.random.seed(seed)
    # Vectorización para eficiencia 
    t_emb = np.random.normal(10, 2, n_samples)      # Ejemplo: media 10ms
    t_search = np.random.uniform(5, 15, n_samples)   # Ejemplo: 5-15ms
    t_llm = np.random.lognormal(3, 0.5, n_samples)   # Cola larga 
    
    return t_emb + t_search + t_llm
