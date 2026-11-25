# Clasificación de textos en Esperanto: judaísmo, cristianismo e islam

Práctica de **Minería de Textos** (curso 2026/27) · Profesor: Javier Paris

**Equipo:** Marco Antonio Benali, Lucía Monteagudo y Sofía Madrazo

---

## ¿De qué va esto?

Queríamos responder a una pregunta sencilla: si te damos un párrafo en esperanto, ¿se puede saber automáticamente si habla de **judaísmo**, de **cristianismo** o de **islam**?

Para conseguirlo hemos hecho todo el recorrido de un proyecto de minería de textos, desde cero:

1. **Conseguir los datos:** los sacamos nosotros mismos de la Wikipedia en esperanto.
2. **Entender el idioma:** construimos nuestro propio "mini spaCy" para esperanto, basado en reglas.
3. **Convertir texto en números:** vectorizamos cada párrafo.
4. **Entrenar y comparar clasificadores:** con scikit-learn.
5. **Probarlo en el mundo real:** clasificamos páginas web descargadas en el momento.

Elegimos el esperanto por un motivo práctico: su gramática es totalmente regular. Los sustantivos acaban en `-o`, los adjetivos en `-a`, los verbos en presente en `-as`... Gracias a eso hemos podido escribir un analizador lingüístico a mano sin usar spaCy, NLTK ni Stanza, que era uno de los requisitos de la práctica.

---

## Cómo está organizado el notebook

Todo el código está en `practicaMineriaSinComentarios.ipynb` y se lee de arriba abajo. Cada celda es un paso.

### 1. Cabecera y librerías
Primero importamos todo lo necesario: `requests` y `BeautifulSoup` para la web, `pandas` y `numpy` para los datos, y `scikit-learn` para los modelos. Después fijamos las constantes del proyecto: la semilla aleatoria (`RANDOM_STATE = 42`), el tamaño del vocabulario (`MAX_FEATURES = 3000`) y los nombres de los ficheros de salida.

### 2. Construcción del dataset (crawler de Wikipedia)
Para cada religión elegimos unos **artículos semilla** (por ejemplo *Torao*, *Sinagogo* o *Talmudo* para el judaísmo) y una lista de **palabras clave**. El crawler funciona así:

- Descarga cada artículo usando la API de Wikipedia.
- Mira sus enlaces internos y solo sigue los que contienen alguna palabra clave del tema, para no acabar leyendo sobre fútbol.
- Para en 10 artículos por categoría y espera 0,2 s entre peticiones para no saturar el servidor.

Cada artículo se trocea en párrafos. Limpiamos el HTML, las referencias tipo `[1]` y las URLs, y descartamos los párrafos de menos de 120 caracteres. Por último quitamos los duplicados, barajamos y guardamos el resultado en `dataset_esperanto_religio.csv` con dos columnas: `text` y `class`.

> El CSV usa `;` como separador, así que sustituimos los `;` que haya dentro del texto por comas.

### 3. Análisis exploratorio
Antes de modelar echamos un vistazo a los datos: cuántos párrafos hay de cada clase, si están equilibradas y cómo varían la longitud del texto, el número de palabras y la longitud media de palabra según la religión. También calculamos la correlación entre estas variables.

### 4. Nuestro modelo NLP para esperanto
Es la parte más artesanal del trabajo. Imita la forma de trabajar de spaCy (`nlp(texto)` devuelve un `Doc` con `Token`s que tienen `lemma_`, `pos_` e `is_stop`), pero todo funciona con reglas propias:

| Pieza | Qué hace |
|---|---|
| **Tokenizador** | Separa el texto en palabras y signos de puntuación con una expresión regular. |
| **Stopwords** | Lista cerrada de palabras vacías del esperanto (`la`, `kaj`, `en`, `mi`...). |
| **Lematizador** | Lleva cada palabra a su forma base mirando la terminación: `preĝas` → `preĝi`, `kristanojn` → `kristano`. |
| **POS-tagger** | Asigna la categoría gramatical por la terminación (`-o` sustantivo, `-a` adjetivo, `-as/-is/-os` verbo...) o por listas cerradas (preposiciones, pronombres...). |
| **Descomposición morfológica** | Separa prefijo, raíz, sufijo y terminación: `kontraŭjuda` → `kontraŭ` + `jud` + `a`. |

La función `nlp.preprocess(texto)` junta todas las piezas: tokeniza, quita la puntuación y las stopwords y devuelve la lista de lemas.

### 5. Vectorización
Primero dividimos los datos en **train (75 %)** y **test (25 %)** de forma estratificada. El vocabulario se construye **solo con train**, para que el modelo no "vea" el test de antemano. Nos quedamos con las 3.000 palabras más frecuentes.

Cada párrafo se convierte en un vector con la técnica **One-Hot mean**: se marca con un 1 cada palabra del vocabulario que aparece y se divide entre el número de palabras del párrafo.

Como comparación también probamos el **Labelling**, que asigna a cada palabra un número y hace la media. Lo incluimos a propósito para enseñar por qué no funciona: esos números son arbitrarios y la media no significa nada.

### 6. Entrenamiento y evaluación
Comparamos cuatro clasificadores de scikit-learn:

- K-Nearest Neighbors (k = 3)
- Regresión logística
- Naive Bayes gaussiano
- Árbol de decisión

Para cada uno calculamos accuracy, precisión, recall y **F1 macro**, y los ordenamos por F1. Repetimos el experimento con los vectores de Labelling para comparar. Del mejor modelo sacamos el *classification report* y la matriz de confusión. Por último hacemos una **validación cruzada K-Fold (5 particiones)** para comprobar que el resultado no depende de una división concreta de los datos.

### 7. Prueba con páginas web reales
Descargamos en directo las páginas de *Kristanismo*, *Islamo* y *Judaismo* de la Wikipedia en esperanto, extraemos sus párrafos y le pedimos al mejor modelo que los clasifique. Mostramos dos resultados por página: la probabilidad media de cada religión párrafo a párrafo y la clase que predice para el texto completo.

### 8. Resultados y conclusiones
Todo se guarda en dos ficheros:

- `predicciones_test.csv`: cada párrafo de test con su clase real, la predicha y si ha acertado.
- `results_religio.json`: un resumen del dataset, el modelo NLP, la comparativa de modelos, el K-Fold y el scraping.

Y para cerrar se imprime un pequeño resumen con el mejor clasificador y su F1.

---

## Cómo ejecutarlo

1. Abre el notebook en Google Colab.
2. Ejecuta las celdas en orden. Las librerías vienen instaladas por defecto en Colab.
3. La primera vez deja `GENERAR_DATASET = True` para descargar los datos. Cuando ya tengas el CSV, cámbialo a `False` y se cargará directamente, mucho más rápido.

**Un aviso:** si haces muchas peticiones seguidas, Wikipedia puede responder con un error `429 Too Many Requests` y alguna categoría se queda sin artículos. Si te pasa, espera unos minutos antes de volver a generar el dataset, o sube la pausa de `time.sleep` en el crawler. Antes de seguir, comprueba en la celda de exploración que aparecen las tres clases.

---

## Estructura del repositorio

```
├── practicaMineriaSinComentarios.ipynb  # notebook con todo el código
├── dataset_esperanto_religio.csv  # dataset generado (text;class)
├── predicciones_test.csv          # predicciones sobre test
├── results_religio.json           # resumen de resultados
└── README.md
```
