# Apuntes TFG: Seguimiento de Múltiples Objetos (MOT)

##  Referencias Principales y Contexto
- **Paper de referencia:** [SportsMOT (arXiv:2304.05170)](https://arxiv.org/pdf/2304.05170) 
  - *Nota (Pág. 1):* "Generally, prevailing state-of-the-art trackers [1,6,11,39, 44]"
- **Lista histórica completa de trackers:** [awesome-multiple-object-tracking (GitHub)](https://github.com/luanshiyinyang/awesome-multiple-object-tracking)
- **Conclusión clave:** Los algoritmos del paradigma **Tracking-by-detection** son los que presentan un mejor funcionamiento general.

##  Trackers Estado del Arte (State-of-the-Art)
- [BOT-SORT](https://github.com/NirAharon/BOT-SORT)
- [OC_SORT](https://github.com/noahcao/OC_SORT)
- [StrongSORT](https://github.com/dyhBUPT/StrongSORT)
- [deep_sort](https://github.com/nwojke/deep_sort)
- [ByteTrack](https://github.com/FoundationVision/ByteTrack)

## Trackers para las Pruebas
### Trackers evaluados en el paper SportsMOT
- CenterTrack
- FairMOT
- QDTrack *(Tracking-by-detection)*
- TransTrack
- GTR
- ByteTrack *(Tracking-by-detection)*
- OC-SORT *(Tracking-by-detection)*
- MixSort-Byte
- MixSort-OC

###  Nuevos Trackers a considerar
- **MOTIP (2025):** - Destaca por su alta popularidad reciente (472 estrellas en su repositorio). 
  - **Justificación:** Es interesante probar un tracker reciente y funcional, especialmente porque se ha demostrado que se puede aplicar a otros dominios más allá del fútbol.

###  Tracker de las prácticas
- DeepEIoU
---

## Clasificación de Trackers según el uso de modelos ReID (Re-Identificación)

> **Nota sobre ByteTrack y ReID:** Aunque la versión original no lo usa, en implementaciones del paradigma Tracking-by-detection, ByteTrack utiliza FastReID para extraer características Re-ID en el dataset MOT17. *(Fuente: [arXiv:2110.06864](https://arxiv.org/pdf/2110.06864) pág. 6)*

##  Clasificación de Trackers según su Modularidad e Intercambio de ReID

###  1. Trackers modulares: Admiten ReID y SÍ se puede intercambiar
Estos rastreadores pertenecen al paradigma *Tracking-by-detection*. Su verdadera identidad y "esencia" reside en sus algoritmos matemáticos de asociación, no en la red que extrae las características visuales. Por tanto, puedes inyectarles o cambiarles el modelo de ReID (como Torchreid) y seguirán siendo ellos mismos.

* **Deep-EIoU:** Sí usa modelo ReID por defecto (como OSNet). 
    * **Se puede intercambiar:** Su esencia no es el modelo visual que usa, sino su innovador algoritmo de expandir iterativamente las cajas delimitadoras (*Iterative Scale-Up ExpansionIoU*). Puedes cambiar OSNet por cualquier otro ReID y Deep-EIoU seguirá siendo Deep-EIoU. **(+2)**
* **ByteTrack:** Originalmente NO utiliza ReID (solo usa movimiento y superposición espacial geométrica). 
    * **Se puede intercambiar/añadir:** Su esencia es su brillante asociación en dos pasos (reciclando detecciones de baja confianza). Es una práctica estándar en investigación inyectarle una matriz de distancias ReID antes de su algoritmo de asociación, manteniendo intacta su identidad lógica. **(+2)**
* **OC-SORT:** Originalmente NO utiliza ReID. 
    * **Se puede intercambiar/añadir:** Su identidad es la corrección matemática que le hace al Filtro de Kalman clásico (*Observation-centric momentum*) para lidiar con movimientos irregulares. Al igual que ByteTrack, puedes acoplarle tu módulo de torchreid como una capa extra de información sin romper la naturaleza del tracker. **(+1)**

---

###  2. Trackers monolíticos (Joint-Detection): SÍ usan ReID explícito pero NO se puede intercambiar
Estos modelos extraen vectores de características visuales, pero su arquitectura está fundida. Intentar cambiar su método de ReID por uno externo destruiría el modelo.

* **FairMOT:** Pertenece al paradigma de "detección y seguimiento conjuntos". Utiliza una rama de ReID acoplada a la misma red neuronal que detecta los objetos, extrayendo las características visuales en un solo paso. 
    * **No se puede intercambiar:** Si le quitas su extractor visual, rompes el *backbone* que también hace las detecciones. Dejaría de ser FairMOT. **(+1)**
* **QDTrack (Quasi-Dense Tracking):** Se apoya fuertemente en la apariencia. Utiliza aprendizaje de similitud casi densa (*Quasi-dense similarity learning*) para comparar visualmente los objetos y asociarlos a lo largo del tiempo. 
    * **No se puede intercambiar:** Esta forma de aprender la similitud visual está profundamente arraigada en cómo se entrena el modelo entero.
* **MixSort-Byte y MixSort-OC:** Estas son las versiones presentadas en el artículo de SportsMOT. Los autores tomaron ByteTrack y OC-SORT y les inyectaron una red llamada MixFormer para actuar como un modelo de asociación basado en la apariencia. 
    * **No se puede intercambiar:** Su esencia es, literalmente, el uso del modelo MixFormer. Si se lo quitas y le pones Torchreid, los devuelves a su estado original (solo ByteTrack u OC-SORT con ReID) y pierden la identidad "MixSort". **(+1)**

---

### 3. Trackers basados en Transformers (ReID/Apariencia implícita): NO se puede intercambiar
Los modelos basados en Transformers no usan una red de ReID separada de la que puedas sacar una matriz de distancias. El mecanismo de "atención" procesa directamente las características visuales para aprender a asociar a los objetos.

* **TransTrack:** Utiliza "consultas de seguimiento" (*track queries*) que aprenden las características visuales del objeto en un fotograma para "preguntar" y buscar a ese mismo objeto en el siguiente fotograma. 
    * **No se puede intercambiar:** La asociación visual ocurre de forma implícita dentro de las capas de atención del Transformer.
* **GTR (Global Tracking Transformers):** Utiliza Transformers para realizar una asociación global a lo largo de muchos fotogramas. Se basa en las representaciones visuales extraídas por la red para discriminar y conectar trayectorias. 
    * **No se puede intercambiar.**
* **MOTIP (Nuevo - 2025):** Formula el seguimiento como un problema de "predicción de ID en contexto". Aunque prescinde de heurísticas clásicas, extrae características visuales (*object-level features*) de los últimos fotogramas y le pide a un Transformer que decida a qué ID pertenece la nueva detección basándose puramente en esa apariencia histórica. 
    * **No se puede intercambiar:** No utiliza cálculos de similitud coseno tradicionales que puedas aislar o sustituir limpiamente.

---

### 4. Trackers que NO usan ReID (Movimiento puro): NO aplica el intercambio modular
Rastreadores que tienen una formulación puramente espacial donde no encaja de forma limpia la inyección de una matriz de ReID sin desvirtuar su propósito.

* **CenterTrack:** Representa a los objetos como puntos centrales. En lugar de usar ReID, toma el fotograma actual y el anterior simultáneamente, y la red neuronal simplemente predice el desplazamiento (*offset espacial*) del punto de un fotograma al otro. 
    * **No aplica:** Su arquitectura está diseñada para prescindir totalmente de la asociación de identidades visuales.