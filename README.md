# deepTSC
Las  series  de  tiempo  están  presentes  en  una  gran  variedad  de  fenómenos.  Sepueden  encontrar  desde  el  análisis  de  mercado  de  valores,  economía,  previsión  de  ventas,hasta la predicción del clima. El tamaño creciente de dichos datos, así como sus característicasde  variabilidad,  alta  dimensionalidad,  correlación  de  características  y  dependencia  temporal,desafían e impulsan el desarrollo y mejora de los métodos de minería de datos en lo referente ala predicción, clasificación e indexación (Amr, 2012).
En particular, abordando la clasificación de series de tiempo (TSC), que básicamente se puededefinir  como  el  problema  de  predecir  las  etiquetas  de  clase  predefinidas  de  las  series  detiempo  Cui  et  al.  (2016),  el  presente  trabajo1propone  la  utilización  de  arquitecturas  híbridasde  aprendizaje  profundo  para  la  clasificación  de  series  de  tiempo  univariadas,  mediante  el acoplamiento de redes neuronales convolucionales (CNN) y long short-term memory (LSTM).
Para  ello  se  implementaron  diez  diversas  arquitecturas  variando  el  orden  y  estructura  de  lascapas recurrentes y convolucionales, bajo enfoques lineales y por raices, buscando obtener elmejor modelo que pudiese competir con otros modelos de la literatura. Finalmente se escogieronlos modelos bilstm_resnet y bilstm_FCN, los cuales tienen como fundamento las arquitecturas ResNet y FCN, sobre los que se pudo determinar un rendimiento superior al de sus pares bajo elenfoque de aprendizaje profundo, y de vanguardia en general, superando a HIVE-COTE y COTE,que presentaban el mejor rendimiento hasta ahora en tareas de TSC.

| **Conjuntos de datos** | **lstm_cnn** | **cnn_bilstm** | **cnn_lstm_separated** | **multi_cnn_bilstm2** | **cnn_lstm_separated2** | **multi_cnn_bilstm** | **resnet_bilstm** | **bilstm_resnet** | **FCN_bilstm** | **bilstm_FCN** |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Adiac | 0.73 | 0.72 | 0.8 | 0.22 | 0.77 | 0.69 | 0.75 | **0.82** | 0.77 | 0.76 |
| Beef | 0.6 | 0.72 | 0.4 | 0.59 | 0.61 | 0.69 | 0.79 | 0.72 | **0.86** | 0.79 |
| CBF | **1.0** | **1.0** | 0.88 | 0.99 | 0.87 | **1.0** | **1.0** | **1.0** | **1.0** | **1.0** |
| ChlorineConcn | **1.0** | **1.0** | **1.0** | **1.0** | **1.0** | **1.0** | **1.0** | **1.0** | **1.0** | **1.0** |
| CinCECGtorso | **1.0** | **1.0** | **1.0** | **1.0** | 0.99 | **1.0** | 0.99 | **1.0** | **1.0** | **1.0** |
| Coffee | **1.0** | **1.0** | 0.97 | 0.94 | **1.0** | 0.88 | **1.0** | **1.0** | **1.0** | **1.0** |
| CricketX | 0.56 | 0.61 | 0.57 | 0.65 | 0.56 | 0.67 | 0.71 | **0.85** | 0.71 | 0.83 |
| CricketY | 0.54 | 0.62 | 0.6 | 0.64 | 0.56 | 0.65 | 0.69 | **0.79** | 0.68 | 0.76 |
| CricketZ | 0.6 | 0.63 | 0.6 | 0.7 | 0.59 | 0.72 | 0.74 | **0.86** | 0.78 | 0.79 |
| DiatomSizeR | **1.0** | 0.99 | 0.99 | 0.95 | 0.99 | 0.99 | **1.0** | **1.0** | 0.97 | **1.0** |
| ECGFiveDays | **1.0** | **1.0** | **1.0** | **1.0** | **1.0** | **1.0** | **1.0** | **1.0** | **1.0** | **1.0** |
| FaceAll | 0.97 | 0.98 | 0.97 | 0.98 | 0.96 | 0.98 | **1.0** | **1.0** | 0.99 | 0.99 |
| FaceFour | 0.76 | 0.81 | 0.85 | 0.87 | 0.82 | 0.87 | **1.0** | **1.0** | **1.0** | **1.0** |
| FacesUCR | 0.97 | 0.97 | 0.97 | 0.97 | 0.96 | 0.98 | 0.99 | **1.0** | 0.99 | **1.0** |
| FiftyWords | 0.6 | 0.62 | 0.63 | 0.63 | 0.61 | 0.65 | 0.6 | **0.82** | 0.66 | 0.81 |
| Fish | 0.88 | 0.82 | 0.86 | 0.74 | 0.84 | 0.83 | **0.97** | **0.97** | 0.91 | 0.96 |
| GunPoint | **1.0** | **1.0** | **1.0** | 0.98 | **1.0** | **1.0** | **1.0** | **1.0** | **1.0** | **1.0** |
| Haptics | 0.32 | 0.38 | 0.31 | 0.32 | 0.32 | 0.32 | 0.2 | **0.52** | 0.27 | 0.51 |
| InlineSkate | 0.3 | 0.33 | 0.3 | 0.29 | 0.32 | 0.21 | 0.19 | 0.41 | 0.22 | **0.49** |
| ItalyPowerDem | 0.92 | 0.91 | 0.92 | 0.92 | 0.92 | 0.92 | 0.93 | **0.95** | 0.94 | 0.93 |
| Lightning2 | **1.0** | **1.0** | **1.0** | **1.0** | **1.0** | **1.0** | **1.0** | **1.0** | **1.0** | **1.0** |
| Lightning7 | 0.65 | 0.73 | 0.55 | 0.66 | 0.54 | 0.64 | 0.81 | **0.89** | 0.8 | 0.86 |
| Mallat | 0.98 | **1.0** | **1.0** | 0.96 | 0.99 | 0.63 | **1.0** | **1.0** | 0.99 | **1.0** |
| MedicalImages | 0.66 | 0.72 | 0.66 | 0.69 | 0.67 | 0.74 | 0.72 | **0.78** | 0.73 | **0.78** |
| MoteStrain | 0.91 | 0.93 | 0.93 | 0.95 | 0.93 | 0.93 | 0.95 | **0.97** | 0.91 | 0.95 |
| NInvFetalECGThx1 | 0.9 | 0.89 | 0.87 | 0.73 | 0.88 | 0.83 | 0.95 | **0.97** | 0.91 | 0.96 |
| NInvFetalECGThx2 | 0.92 | 0.91 | 0.92 | 0.89 | 0.9 | 0.89 | 0.93 | **0.95** | 0.93 | **0.95** |
| OliveOil | 0.87 | 0.88 | **0.93** | 0.61 | 0.85 | 0.72 | 0.37 | 0.0 | 0.5 | 0.0 |
| OSULeaf | 0.4 | 0.46 | 0.49 | 0.49 | 0.44 | 0.54 | 0.7 | **0.99** | 0.65 | 0.9 |
| SonyAIBORobotS1 | **1.0** | 0.99 | 0.99 | **1.0** | 0.99 | **1.0** | **1.0** | **1.0** | **1.0** | **1.0** |
| SonyAIBORobotS2 | 0.99 | 0.96 | 0.99 | 0.98 | 0.98 | 0.99 | **1.0** | **1.0** | 0.99 | **1.0** |
| StarlightCurves | 0.95 | 0.94 | 0.93 | 0.94 | 0.93 | 0.83 | 0.94 | 0.95 | 0.92 | **0.96** |
| SwedishLeaf | 0.89 | 0.88 | 0.89 | 0.82 | 0.89 | 0.93 | **0.97** | 0.96 | **0.97** | 0.95 |
| Symbols | 0.98 | 0.96 | 0.97 | 0.96 | 0.96 | 0.98 | 0.93 | **0.99** | 0.95 | 0.98 |
| SyntheticControl | **1.0** | 0.99 | 0.99 | 0.96 | 0.99 | 0.97 | 0.99 | 0.99 | 0.99 | 0.99 |
| Trace | **1.0** | **1.0** | **1.0** | **1.0** | **1.0** | **1.0** | **1.0** | **1.0** | **1.0** | **1.0** |
| TwoLeadECG | **1.0** | **1.0** | **1.0** | **1.0** | **1.0** | **1.0** | **1.0** | **1.0** | **1.0** | **1.0** |
| TwoPatterns | **1.0** | **1.0** | **1.0** | **1.0** | **1.0** | **1.0** | **1.0** | **1.0** | **1.0** | **1.0** |
| UWaveGestureLibX | **0.81** | 0.79 | 0.79 | 0.76 | 0.8 | 0.8 | 0.79 | 0.79 | 0.76 | 0.8 |
| UWaveGestureLibY | 0.72 | 0.68 | 0.69 | 0.65 | 0.68 | 0.7 | 0.65 | **0.73** | 0.62 | **0.73** |
| UWaveGestureLibZ | 0.73 | 0.72 | 0.7 | 0.68 | 0.71 | 0.71 | 0.62 | 0.73 | 0.66 | **0.77** |
| Wafer | **1.0** | **1.0** | **1.0** | **1.0** | **1.0** | **1.0** | **1.0** | **1.0** | **1.0** | **1.0** |
| WordSynonyms | 0.64 | 0.66 | 0.66 | 0.69 | 0.63 | 0.66 | 0.73 | **0.83** | 0.7 | 0.81 |
| Yoga | 0.91 | 0.87 | 0.82 | 0.78 | 0.81 | 0.83 | 0.88 | **0.97** | 0.87 | 0.96 |
| **Ganados** | **15** | **12** | **11** | **9** | **9** | **11** | **18** | **36** | **15** | **23** |
| **Ranking Promedio** | **5.886** | **6.090** | **6.465** | **7.125** | **7.022** | **6.079** | **4.909** | **2.920** | **5.227** | **3.272** |
| | (5) | (7) | (8) | (10) | (9) | (6) | (3) | (1) | (4) | (2) |
