# Taller 3: Simulación Montecarlo - Evaluación de Infraestructura

Este repositorio contiene una librería de Python orientada a la simulación estadística. Aplica los fundamentos del Método de Montecarlo y técnicas de muestreo avanzado para la evaluación de riesgos en arquitecturas RAG, pipelines de datos y tráfico de servidores.

## Estructura del Repositorio

El proyecto está estructurado como un paquete instalable de Python:

* `notebooks/Evaluacion_Simulacion.ipynb`: Cuaderno de Jupyter con los análisis gráficos y conclusiones.
* `simulacion/`: Código fuente de la librería.
  * `latencia.py`: Lógica del Caso 1 (Montecarlo Estándar).
  * `importancia.py`: Lógica del Caso 2 (Importance Sampling).
  * `rechazo.py`: Lógica del Caso 3 (Rejection Sampling).

---

## Casos de Estudio y Resultados

### Caso 1: Análisis Estocástico de Latencia en Arquitecturas RAG
Se simuló el tiempo total de respuesta combinando distribuciones Normal (Embeddings), Uniforme (Búsqueda) y Lognormal (Inferencia LLM) para evaluar un SLA estricto de 50 ms.
* **Resultado:** El percentil 99 se ubicó en **84.66 ms**, incumpliendo el SLA.
* **Conclusión:** La naturaleza Lognormal del LLM genera "colas pesadas". Se recomienda aplicar técnicas de cuantización de modelos o implementar caché semántica para reducir la varianza.

### Caso 2: Simulación de Eventos Raros (Importance Sampling)
Se estimó la probabilidad de que un proceso crítico supere un límite de tiempo (fallo), utilizando una distribución propuesta para forzar el evento raro y reducir la varianza.
* **Resultado:** Probabilidad de fallo estimada en **8.20%**. El factor de reducción de varianza fue de **0.49x**.
* **Conclusión:** Al ser el factor menor a 1, la técnica fue menos eficiente que el muestreo aleatorio simple. La distribución propuesta elegida es subóptima y aumentó el ruido en el sistema, por lo que requiere rediseño.

### Caso 3: Simulación de Tráfico Bimodal (Rejection Sampling)
Se generaron muestras sintéticas para replicar fluctuaciones de carga diurnas (dos picos de tráfico) utilizando una distribución envolvente y un criterio de aceptación.
* **Resultado:** El modelo replicó visualmente la curva teórica con éxito, pero con una Tasa de Aceptación de solo **16.67%**.
* **Conclusión:** Existe una alta ineficiencia computacional, descartando más del 80% de los ciclos de CPU. Se recomienda ajustar la constante de escala o usar una envolvente más ceñida.

---

## Instalación y Uso

Para ejecutar este proyecto localmente, se recomienda utilizar un entorno virtual (venv):

1. **Crear y activar el entorno virtual:**
   ```bash
   python -m venv .venv
   # En Windows:
   .venv\Scripts\activate
   # En Linux/Mac:
   source .venv/bin/activate
