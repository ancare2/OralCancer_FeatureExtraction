# OralCancer_FeatureExtraction

Este repositorio contiene el código y los recursos utilizados en el Trabajo de Fin de Máster (TFM) centrado en el análisis de imágenes médicas de cáncer oral mediante técnicas de segmentación y extracción de características radiómicas.

## Objetivo

El proyecto tiene como objetivo explorar patrones internos en imágenes de lesiones orales, identificando subgrupos a partir de características cuantitativas y análisis de clustering, y proporcionando herramientas para la caracterización más detallada de las imágenes.

## Contenido

- Segmentación: Se implementaron modelos de segmentación, incluyendo U-Net y SegFormer, para aislar de manera automática las regiones de interés en las imágenes.

- Extracción de características radiómicas: Se calcularon variables relacionadas con intensidad, textura y heterogeneidad de las imágenes, permitiendo una descripción cuantitativa de las regiones segmentadas.

- Análisis de clustering: Se aplicaron algoritmos como K-Means, Gaussian Mixture Models (GMM), Spectral Clustering y Agglomerative Clustering para identificar subgrupos dentro de los datos. Se evaluó la cohesión y separación mediante métricas como el Silhouette Score.

- Visualización: Se generaron scatter plots, box plots y ejemplos de imágenes representativas por clúster para facilitar la interpretación de los resultados.

- Selección de características clave: Se identificaron las variables más relevantes para la diferenciación de clústeres, mostrando cómo ciertas propiedades radiómicas destacan repetidamente en distintos métodos de agrupamiento.

## Requisitos

Python 3.9 o superior

Bibliotecas principales: numpy, pandas, matplotlib, opencv-python, scikit-learn, torch (para modelos de segmentación)

## Uso

- Preparar los datos: Colocar las imágenes originales en la carpeta Segmentacion/segmentacion_resultados.

- Ejecutar segmentación: Correr los scripts de U-Net o SegFormer para obtener máscaras de segmentación.

- Extraer características radiómicas: Ejecutar los scripts de extracción sobre las regiones segmentadas.

- Análisis y visualización: Correr los notebooks para clustering, generación de gráficos y selección de características relevantes.

## Contribuciones

Este repositorio está diseñado como apoyo académico. Para sugerencias, mejoras o dudas, se pueden abrir issues o contactar a la autora.
