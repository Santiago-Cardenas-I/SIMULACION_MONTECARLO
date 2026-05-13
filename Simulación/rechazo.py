def rejection_sampling(target_func, proposal_dist, k, n_samples=10000):
    """Genera muestras siguiendo la distribución bimodal[cite: 22]."""
    samples = []
    while len(samples) < n_samples:
        z = proposal_dist.rvs(1) # Muestra de q(x)
        u = np.random.uniform(0, k * proposal_dist.pdf(z))
        if u <= target_func(z):
            samples.append(z)
    return np.array(samples)