# Segmentación de Clientes Mayoristas con Aprendizaje No Supervisado

**Trabajo Práctico Final — Módulo: Aprendizaje No Supervisado**

**Autores:** Federico Moran · Juan Barboza


[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1-Lt_lY2_Pml4ACgvVstlCKixHlTx3WGN?usp=sharing)

## Descripción

Aplicación de técnicas de aprendizaje no supervisado sobre el dataset
[Wholesale Customers (UCI Machine Learning Repository)](https://archive.ics.uci.edu/dataset/292/wholesale%2Bcustomers)
para segmentar clientes mayoristas según su gasto anual en seis categorías de producto.

El dataset contiene **440 observaciones** y **8 variables**: `Channel` (Horeca / Retail),
`Region` (Lisboa / Oporto / Otra) y el gasto anual en `Fresh`, `Milk`, `Grocery`,
`Frozen`, `Detergents_Paper` y `Delicassen`.

## Metodología

| Etapa | Técnica |
|---|---|
| Baseline (solo referencia) | K-Means |
| Clustering 1 | Agglomerative Clustering |
| Clustering 2 | Gaussian Mixture Models |
| Clustering avanzado | Deep Embedded Clustering (DEC) |
| Visualización | t-SNE |
| Evaluación interna | Silhouette, Davies-Bouldin, Calinski-Harabasz |
| Evaluación externa | vs. `Channel` como pseudo-etiqueta (ARI, NMI) |

## Estructura del repositorio

```
├── notebooks/
│   └── TPF_NoSupervisado_Moran_Barboza.ipynb   # Notebook principal (autocontenido)
├── data/          # Dataset (el notebook lo descarga automáticamente desde UCI)
├── figures/       # Figuras exportadas para el informe y la presentación
├── docs/          # Informe técnico (GitHub Pages)
└── requirements.txt
```

## Reproducibilidad

El notebook es **autocontenido**: descarga el dataset directamente desde la URL pública
de UCI (ZIP → CSV), por lo que no requiere ningún archivo local previo.

**Opción A — Google Colab:** abrir con el badge de arriba y ejecutar todas las celdas.

**Opción B — Local:**

```bash
python -m venv .venv
.venv\Scripts\activate        # Windows  (source .venv/bin/activate en Linux/Mac)
pip install -r requirements.txt
jupyter notebook notebooks/TPF_NoSupervisado_Moran_Barboza.ipynb
```

Se fijan semillas aleatorias (`random_state=42`) en todos los algoritmos estocásticos.

## Informe

El informe técnico está disponible en [`docs/index.md`](docs/index.md) y publicado
como sitio en GitHub Pages: `https://fedemoranf-alt.github.io/TPF-NoSupervisado-Moran_Barboza/` 

## Referencias

- Cardoso, M. (2013). *Wholesale customers* [Dataset]. UCI Machine Learning Repository. https://doi.org/10.24432/C5030X
- Xie, J., Girshick, R., & Farhadi, A. (2016). *Unsupervised Deep Embedding for Clustering Analysis* (DEC). ICML 2016.
- van der Maaten, L., & Hinton, G. (2008). *Visualizing Data using t-SNE*. JMLR 9.
