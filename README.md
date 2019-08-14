# Clasificación de series de tiempo por medio de arquitecturas híbridas de aprendizaje profundo
Las  series  de  tiempo  están  presentes  en  una  gran  variedad  de  fenómenos.  Se pueden  encontrar  desde  el  análisis  de  mercado  de  valores,  economía,  previsión  de  ventas,hasta la predicción del clima. El tamaño creciente de dichos datos, así como sus característicasde  variabilidad,  alta  dimensionalidad,  correlación  de  características  y  dependencia  temporal, desafían e impulsan el desarrollo y mejora de los métodos de minería de datos en lo referente ala predicción, clasificación e indexación (Amr, 2012).

En particular, abordando la clasificación de series de tiempo (TSC), que básicamente se puede definir  como  el  problema  de  predecir  las  etiquetas  de  clase  predefinidas  de  las  series  detiempo  Cui  et  al.  (2016),  el  presente  trabajo1propone  la  utilización  de  arquitecturas  híbridasde  aprendizaje  profundo  para  la  clasificación  de  series  de  tiempo  univariadas,  mediante  el acoplamiento de redes neuronales convolucionales (CNN) y long short-term memory (LSTM).

Para  ello  se  implementaron  diez  diversas  arquitecturas  variando  el  orden  y  estructura  de  lascapas recurrentes y convolucionales, bajo enfoques lineales y por raices, buscando obtener elmejor modelo que pudiese competir con otros modelos de la literatura. Finalmente se escogieronlos modelos bilstm_resnet y bilstm_FCN, los cuales tienen como fundamento las arquitecturas ResNet y FCN, sobre los que se pudo determinar un rendimiento superior al de sus pares bajo elenfoque de aprendizaje profundo, y de vanguardia en general, superando a HIVE-COTE y COTE, que presentaban el mejor rendimiento hasta ahora en tareas de TSC.

## Resultados de los modelos implementados
A continuación se presentan los resultados generales para todos los modelos implementados y los conjuntos de datos bajo la métrica del accuracy, con una validación cruzada estratificada de 5-fold.

| **Conjuntos de datos** | **lstm_cnn** | **cnn_bilstm** | **cnn_lstm_separated** | **multi_cnn_bilstm2** | **cnn_lstm_separated2** | **multi_cnn_bilstm** | **resnet_bilstm** | **bilstm_resnet** | **FCN_bilstm** | **bilstm_FCN** |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Adiac | 0.74 | 0.73 | 0.8 | 0.24 | 0.78 | 0.7 | 0.75 | **0.86** | 0.78 | 0.83 |
| Beef | 0.69 | 0.78 | 0.52 | 0.68 | 0.69 | 0.76 | 0.83 | 0.79 | **0.89** | 0.83 |
| CBF | **1.0** | **1.0** | 0.92 | 0.99 | 0.92 | **1.0** | **1.0** | **1.0** | **1.0** | **1.0** |
| ChlorineConc | **1.0** | **1.0** | **1.0** | **1.0** | **1.0** | **1.0** | **1.0** | **1.0** | **1.0** | **1.0** |
| CinCECGtorso | **1.0** | **1.0** | **1.0** | **1.0** | **1.0** | **1.0** | 0.99 | **1.0** | **1.0** | **1.0** |
| Coffee | **1.0** | **1.0** | 0.99 | 0.97 | **1.0** | 0.94 | **1.0** | **1.0** | **1.0** | **1.0** |
| CricketX | 0.6 | 0.65 | 0.6 | 0.68 | 0.6 | 0.7 | 0.74 | **0.85** | 0.73 | 0.84 |
| CricketY | 0.58 | 0.65 | 0.63 | 0.67 | 0.6 | 0.68 | 0.71 | **0.83** | 0.71 | 0.83 |
| CricketZ | 0.63 | 0.66 | 0.63 | 0.73 | 0.63 | 0.74 | 0.76 | **0.86** | 0.79 | 0.74 |
| DiatomSizeRed | **1.0** | 0.99 | 0.99 | 0.97 | 0.99 | 0.99 | **1.0** | **1.0** | 0.98 | **1.0** |
| ECGFiveDays | **1.0** | **1.0** | **1.0** | **1.0** | **1.0** | **1.0** | **1.0** | **1.0** | **1.0** | **1.0** |
| FaceAll | 0.98 | 0.98 | 0.98 | 0.98 | 0.97 | 0.98 | **1.0** | **1.0** | 0.99 | **1.0** |
| FaceFour | 0.83 | 0.86 | 0.9 | 0.91 | 0.87 | 0.91 | **1.0** | **1.0** | **1.0** | **1.0** |
| FacesUCR | 0.98 | 0.97 | 0.97 | 0.98 | 0.96 | 0.98 | 0.99 | **1.0** | 0.99 | **1.0** |
| FiftyWords | 0.62 | 0.63 | 0.65 | 0.65 | 0.63 | 0.66 | 0.62 | **0.81** | 0.68 | **0.81** |
| Fish | 0.9 | 0.85 | 0.88 | 0.79 | 0.87 | 0.86 | **0.97** | **0.97** | 0.92 | 0.96 |
| GunPoint | **1.0** | **1.0** | **1.0** | 0.99 | **1.0** | **1.0** | **1.0** | **1.0** | **1.0** | **1.0** |
| Haptics | 0.46 | 0.5 | 0.45 | 0.46 | 0.46 | 0.46 | 0.37 | **0.48** | 0.42 | 0.32 |
| InlineSkate | 0.41 | 0.43 | 0.4 | 0.39 | 0.42 | 0.33 | 0.31 | 0.51 | 0.33 | **0.56** |
| ItalyPowerDemand | 0.96 | 0.96 | 0.96 | 0.96 | 0.96 | 0.96 | **0.97** | **0.97** | **0.97** | **0.97** |
| Lightning2 | **1.0** | **1.0** | **1.0** | **1.0** | **1.0** | **1.0** | **1.0** | **1.0** | **1.0** | **1.0** |
| Lightning7 | 0.71 | 0.78 | 0.63 | 0.72 | 0.62 | 0.7 | 0.84 | **0.91** | 0.84 | 0.88 |
| Mallat | 0.99 | **1.0** | **1.0** | 0.96 | 0.99 | 0.67 | **1.0** | **1.0** | 0.99 | **1.0** |
| MedicalImages | 0.78 | 0.82 | 0.78 | 0.81 | 0.79 | 0.83 | 0.8 | 0.84 | 0.82 | **0.85** |
| MoteStrain | 0.96 | 0.97 | 0.97 | 0.97 | 0.97 | 0.96 | **0.98** | **0.98** | 0.96 | **0.98** |
| NInvFetalECGThx1 | 0.91 | 0.89 | 0.87 | 0.74 | 0.88 | 0.83 | 0.93 | **0.97** | 0.91 | 0.96 |
| NInvFetalECGThx2 | 0.93 | 0.92 | 0.92 | 0.89 | 0.9 | 0.89 | 0.94 | **0.95** | 0.94 | 0.94 |
| OliveOil | 0.9 | 0.91 | **0.94** | 0.71 | 0.89 | 0.79 | 0.56 | 0.91 | 0.67 | 0.88 |
| OSULeaf | 0.51 | 0.56 | 0.58 | 0.59 | 0.54 | 0.62 | 0.75 | **0.99** | 0.71 | 0.97 |
| SonyAIBORobotSurf1 | **1.0** | 0.99 | 0.99 | **1.0** | 0.99 | **1.0** | **1.0** | **1.0** | **1.0** | **1.0** |
| SonyAIBORobotSurf2 | 0.99 | 0.98 | 0.99 | 0.99 | 0.99 | 0.99 | **1.0** | **1.0** | **1.0** | **1.0** |
| StarlightCurves | 0.97 | 0.97 | 0.96 | 0.96 | 0.92 | 0.91 | 0.97 | 0.97 | 0.95 | **0.98** |
| SwedishLeaf | 0.9 | 0.89 | 0.9 | 0.83 | 0.9 | 0.93 | **0.97** | 0.95 | **0.97** | 0.95 |
| Symbols | 0.98 | 0.97 | 0.97 | 0.97 | 0.97 | 0.98 | 0.94 | **0.99** | 0.95 | **0.99** |
| SyntheticControl | **1.0** | 0.99 | 0.99 | 0.97 | 0.99 | 0.98 | 0.99 | 0.99 | 0.99 | 0.99 |
| Trace | **1.0** | **1.0** | **1.0** | **1.0** | **1.0** | **1.0** | **1.0** | **1.0** | **1.0** | **1.0** |
| TwoLeadECG | **1.0** | **1.0** | **1.0** | **1.0** | **1.0** | **1.0** | **1.0** | **1.0** | **1.0** | **1.0** |
| TwoPatterns | **1.0** | **1.0** | **1.0** | **1.0** | **1.0** | **1.0** | **1.0** | **1.0** | **1.0** | **1.0** |
| UWaveGestureLibX | 0.83 | 0.82 | 0.82 | 0.79 | 0.82 | 0.82 | 0.82 | 0.82 | 0.79 | **0.83** |
| UWaveGestureLibY | 0.75 | 0.72 | 0.73 | 0.69 | 0.72 | 0.74 | 0.69 | 0.74 | 0.66 | **0.79** |
| UWaveGestureLibZ | 0.76 | 0.75 | 0.74 | 0.72 | 0.74 | 0.75 | 0.67 | 0.74 | 0.70 | **0.80** |
| Wafer | **1.0** | **1.0** | **1.0** | **1.0** | **1.0** | **1.0** | **1.0** | **1.0** | **1.0** | **1.0** |
| WordSynonyms | 0.68 | 0.7 | 0.69 | 0.72 | 0.67 | 0.69 | 0.75 | **0.82** | 0.73 | **0.82** |
| Yoga | 0.96 | 0.93 | 0.91 | 0.89 | 0.9 | 0.92 | 0.94 | **0.98** | 0.94 | **0.98** |
| **Ganados** | **14** | **12** | **11** | **9** | **10** | **11** | **20** | **34** | **17** | **30** | 
| **Ranking Promedio** | **5.807** | **5.931** | **6.557** | **7.170** | **6.955** | **6.261** | **4.886** | **3.057** | **5.182** | **3.193** |
| | (5) | (6) | (8) | (10) | (9) | (7) | (3) | (1) | (4) | (2) |

Finalmente a partir de todos los resultados presentados anteriormente se seleccionan los modelos bilstm_resnet y bilstm_FCN para realizar las respectivas comparaciones con otros modelos de la literatura.

## Comparación con otros enfoques de deep learning
A continuación se presentan los resultados generales de las diferencias críticas para todos los modelos y los conjuntos de datos bajo la métrica del accuracy, obtenidos del estudio de Fawaz et al. (2019), queagrupa la revisión de los últimos modelos de la literatura enfocados en el aprendizaje profundo.

![alt text](https://github.com/danyrubiano/deepTSC/blob/master/Images/CD_nm_deep.png)

A  partir  de  ello,  se  puede  afirmar  la  supremacía  en  rendimiento  bajo  la  métrica del  accuracy  que  alcanzaron  a  nivel  general  los  modelos  híbridos  de  aprendizaje  profundo implementados en el presente trabajo, lo que refleja el poder que presenta la combinación de arquitecturas convolucionales y recurrentes para afrontar los problemas de TSC, en comparación con enfoques puramente convolucionales o puramente recurrentes.

## Comparación con otros enfoques de la literatura
A continuación se presentan los resultados generales bajo la métrica del accuracy para varios modelos con diversos enfoques, obtenidos del estudio de (Lines et al., 2018), en con-junto con los modelos aquí propuestos.

![alt text](https://github.com/danyrubiano/deepTSC/blob/master/Images/CD_nm_acc.png)

Para concluir, se afirmar decir que los modelos híbridos de aprendizaje profundo implementados en el presente trabajo, alcanzaron un rendimiento de vanguardia en los problemas de TSC, superando a modelos mucho más complicados como HIVE-COTE y COTE,lo  que  refleja  el  poder  que  puede  llegar  a  presentar  las  arquitecturas  híbridas  de  aprendizaje profundo, específicamente la combinación entre CNN Y LSTM, en la clasificación de series detiempo.

## Referencias
Amr, T. (2012). Survey on time-series data classification.TSDM 2012.

Cui, Z., Chen, W., & Chen, Y. (2016).  Multi-scale convolutional neural networks for time series classification.CoRR,abs/1603.06995.

Fawaz, H. I., Forestier, G., Weber, J., Idoumghar, L., & Muller, P.-A. (2019). Deep learning for timeseries classification: a review. Data Mining and Knowledge Discovery,33, 917–963.

Lines,  J.,  Taylor,  S.,  &  Bagnall,  A.  (2018). Hive-cote:  The  hierarchical  vote  collective  oftransformation-based ensembles for time series classification. ACM Transactions on KnowledgeDiscovery from Data (TKDD),12 (52).
