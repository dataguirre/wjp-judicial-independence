# WJP Independencia Judicial

Pipeline de NLP para detectar y analizar eventos relacionados con la independencia judicial (IJ) a partir de resúmenes de noticias por país, construido sobre el marco del [World Justice Project](https://worldjusticeproject.org/).

![Dashboard](assets/wjp_ji_interface.png)


---

## Instalación

**Requisitos**: [`uv`](https://docs.astral.sh/uv/). 

`uv` es un manejador de paquetes y ambientes virtuales **moderno**. Aquí puede encontrar una guía de instalación: https://docs.astral.sh/uv/getting-started/installation/

**Solo resultados** (sin dependencias del pipeline):
Con esta instalación mínima, podrá ejecutar el notebook del módulo 3 (`notebooks/module3_visualization_and_analytics`) o su versión de prototipo funcional con `streamlit`

```bash
git clone https://github.com/dataguirre/wjp-judicial-independence.git
cd wjp-judicial-independence # entrar al proyecto
uv sync # agregar dependencias mínimas
streamlit run app.py # ejecutar dashboard con resultados
```
- NOTA: Para realizar una reproducción completa de los datos, diríjase a la sección de Reproducibilidad. Esto descargará librerías más pesadas como torch, transformers, nvidia drivers, etc., para poder correr modelos de lenguaje de forma local (se utilizó una GPU RTX 3090)
- 
## Estructura del proyecto

```
├── src/wjp_judicial_independence/
│   ├── config.py           # Configuración de rutas
│   ├── preprocessing.py    # Extracción de eventos
│   ├── classifier.py       # Módulo 1: clasificación binaria
│   ├── sentiment.py        # Módulo 2a: clasificación de sentimiento
│   ├── analysis.py         # Comparación de estrategias
│   ├── plot.py             # Todas las visualizaciones
│   └── utils.py            # Despacho de API y reintentos
├── notebooks/              # Notebooks exploratorios (módulos 1–3)
│   ├── module1_classification.py                # estrategias de clasificacion de noticias de independencia judicial
│   ├── module2_sentiment.py                     # clasificacion de sentimiento (amenaza/neutral/fortalecimiento) de independencia judicial
│   ├── module2_topic_modelling.py               # modelamiento de temas de independencia judicial
│   └── module3_visualization_and_analysis.py    # Visualización y análisis de resultados
├── scripts/
│   ├── pipeline.py                       # Pipeline de extremo a extremo (equivalente
│   └── precompute_topics_per_class.py    # Caché de artefactos para el dashboard
├── data/
│   ├── raw/                # Archivos JSON por país
│   └── interim/            # Salidas del pipeline/notebooks (Parquet + JSON)
├── assets/                 # Recursos estáticos (logos, capturas)
└── app.py                  # Dashboard Streamlit
```
## Descripción general

El pipeline procesa resúmenes de noticias estructurados para tres países (Hungría, Italia, Polonia) y produce:

1. **Módulo 1** — Clasificación binaria: ¿es cada evento relevante para la independencia judicial?
2. **Módulo 2a** — Clasificación de sentimiento: ¿el evento *amenaza* o *fortalece* la independencia judicial, o es neutral?
3. **Módulo 2b** — Modelado de temas (BERTopic): ¿cuáles son los principales temas dentro de los eventos relevantes?
4. **Dashboard** — Aplicación Streamlit interactiva para explorar resultados por estrategia, país y tema.

---


## Enfoque metodológico

### Extracción de eventos

Los datos crudos son archivos JSON por país donde los resúmenes de noticias están anidados por pilar WJP y categoría de impacto (Muy Positivo → Muy Negativo). Los eventos se extraen a nivel de párrafo filtrando líneas que siguen la convención markdown `* **`, que identifica de forma fiable descripciones de eventos individuales y excluye títulos, conclusiones y metadatos editoriales.

### Clasificación (Módulo 1)

Se implementan tres estrategias de clasificación en paralelo para comparar su robustez:

| Estrategia | Método | Coste | Velocidad |
|------------|--------|-------|-----------|
| `embeddings` | Similitud coseno contra descripciones de categorías de referencia con `all-mpnet-base-v2` | Gratuito | ~2 min |
| `llm` | Inferencia local con Qwen2.5-7B-Instruct (cuantizado a 4 bits) | Gratuito (GPU) | ~4 min |
| `llm-api` | Inferencia vía API con GPT-4o-mini o Claude | ~$2–3 por 1K eventos | ~10–15 min |

Las tres comparten el mismo diseño de prompt: un mensaje de sistema con 10 criterios de inclusión y 5 de exclusión basados en las definiciones del WJP (p. ej., *"Un arresto por drogas NO es independencia judicial. Un tribunal que revoca un decreto del gobierno SÍ lo es."*).

### Clasificación de sentimiento (Módulo 2a)

Para los eventos marcados como relevantes para IJ, la misma arquitectura de tres estrategias asigna una de tres etiquetas — **amenaza**, **fortalecimiento** o **neutral** — a partir de definiciones derivadas de la literatura sobre IJ. El sentimiento se desacopla intencionalmente del encuadre mediático: una noticia con framing negativo puede representar un fortalecimiento si los tribunales actuaron de forma independiente.

### Modelado de temas (Módulo 2b)

Se utiliza BERTopic para el descubrimiento interpretable de temas:

- **Embeddings**: `all-mpnet-base-v2`
- **Reducción dimensional**: UMAP (`n_components=10, metric=cosine`)
- **Clustering**: HDBSCAN (`min_cluster_size=5`)
- **Representación**: ClassTF-IDF → KeyBERT + Maximal Marginal Relevance

Se entrenan dos niveles de modelos: un **modelo general** (8 temas sobre todos los eventos) y **modelos por país** para una exploración más granular. Los temas se estratifican por sentimiento y pilar WJP mediante `topics_per_class` de BERTopic.

### Pre-computación

Los artefactos de BERTopic (DataFrames de temas, figuras Plotly, mapas de color) se pre-computan una sola vez mediante `scripts/precompute_topics_per_class.py` y se cachean en `data/interim/module3/` como Parquet y JSON. El dashboard los carga directamente sin necesidad de modelos ML en tiempo de ejecución.

---

## Decisiones clave

**Tres estrategias en lugar de una.** Ejecutar embeddings, LLM local y LLM API en paralelo permite analizar el acuerdo entre estrategias: las discrepancias señalan casos ambiguos que merecen revisión. También permite al usuario elegir su propio equilibrio entre coste y calidad.

**Relevancia y sentimiento desacoplados.** Clasificar *si* un evento es relevante para IJ de forma separada a *cómo* la afecta evita confundir el encuadre mediático con el impacto judicial. Un evento de noticias "Muy Negativo" puede fortalecer la independencia judicial.

**Cuantización local.** La inferencia a 4 bits mediante BitsAndBytes hace accesible la estrategia LLM local sin una GPU de alta gama, reduciendo la barrera para la reproducibilidad offline completa.

**Clasificación basada en prompts sin fine-tuning.** La adaptación al dominio se codifica en el prompt del sistema en lugar de en los pesos del modelo, lo que facilita revisar las definiciones sin necesidad de reentrenar. Los prompts incorporan directamente el marco conceptual del WJP.

**Pre-computación en lugar de inferencia bajo demanda.** Separar el pipeline computacionalmente costoso del dashboard interactivo garantiza tiempos de carga de menos de un segundo y elimina las dependencias de modelos de la aplicación desplegada.

---

## Limitaciones

**Dataset pequeño.** El pipeline procesa ~1.200 eventos de tres países. Los resultados son exploratorios y no son estadísticamente generalizables. Añadir países o períodos de tiempo requeriría re-ejecutar el pipeline completo.

**Embeddings basados en referencias.** La estrategia de embeddings depende de tres descripciones de referencia construidas manualmente. Su cobertura y el umbral empleado (0,5) no han sido validados contra expertos del dominio ni etiquetas anotadas. Distintas descripciones de referencia o umbrales pueden cambiar sustancialmente los resultados.

**Sin benchmark de evaluación humana.** Ninguna de las tres estrategias ha sido calibrada contra etiquetas anotadas por humanos. El acuerdo entre estrategias proporciona una verificación de consistencia, pero no una estimación de precisión contra verdad de terreno.

**Heurística de extracción de eventos.** El filtro de párrafos `* **` es específico al formato de los archivos JSON de origen. Si el formato de los datos cambia, la extracción fallará de forma silenciosa.

**Fragilidad de los prompts.** Las estrategias basadas en LLM analizan las salidas mediante búsqueda de subcadenas (`"1"`, `"amenaza"`, `"fortalecimiento"`). Salidas parafraseadas o inesperadas del modelo se asignan por defecto a la clase negativa o neutral sin notificación.

**Hiperparámetros del modelo de temas.** La configuración de UMAP y HDBSCAN se eligió por inspección, no mediante búsqueda sistemática. El número de temas del modelo general (8) está fijado. Los resultados pueden variar con diferentes semillas o configuraciones.

**Alcance temporal y geográfico.** Los datos cubren Hungría, Italia y Polonia sin un rango de fechas explícito. Las comparaciones entre países o temporales deben interpretarse con cautela dadas las diferencias en la cobertura de las fuentes.

---

## Reproducibilidad

**Pipeline completo** (incluye LLM local, BERTopic, clientes API):

```bash
uv sync --extra pipeline
uv run python scripts/pipeline.py --strategies embeddings llm-api --api-provider openai --api-key <KEY>
uv run python scripts/precompute_topics_per_class.py
streamlit run app.py
```

### Opciones del pipeline

| Flag | Descripción |
|------|-------------|
| `--strategies` | Una o más de: `embeddings`, `llm`, `llm-api` |
| `--api-provider` | `openai` o `anthropic` (requerido para `llm-api`) |
| `--api-key` | Clave API del proveedor elegido |
| `--api-model` | Nombre del modelo (p. ej. `gpt-4o-mini`, `claude-3-5-haiku-latest`) |
| `--force` | Re-ejecutar pasos aunque la salida ya exista |
