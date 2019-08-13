import numpy as np
import pandas as pd
from keras.models import model_from_yaml
from keras.models import load_model
from sklearn import model_selection
from numpy import argmax

import get_history
import get_stats
import load_datasets
import load_model
import plot_distribution
import m_FCN #1
import m_FCN_bilstm #1
import m_FCN_bilstm_separated #2
import m_bilstm_FCN #1
import m_resnet #1
import m_resnet_bilstm #1
import m_resnet_bilstm_separated #2
import m_bilstm_resnet #1
import m_cnn #1
import m_cnn_lstm #1
import m_cnn_bilstm #1
import m_cnn_lstm_separated #2
import m_cnn_lstm_separated2 #3
import m_multi_cnn #3
import m_multi_cnn_lstm #3
import m_multi_cnn_bilstm #3
import m_multi_cnn_bilstm2 #2
import m_lstm_cnn #1


direc = 'UCR'
datasets = ['Adiac','Beef','CBF','ChlorineConcentration','CinCECGtorso','Coffee','CricketX','CricketY','CricketZ','DiatomSizeReduction','ECGFiveDays','FaceAll','FaceFour','FacesUCR','FiftyWords','Fish','GunPoint','Haptics','InlineSkate','ItalyPowerDemand','Lightning2','Lightning7','Mallat','MedicalImages','MoteStrain','NonInvasiveFetalECGThorax1','NonInvasiveFetalECGThorax2','OliveOil','OSULeaf','SonyAIBORobotSurface1','SonyAIBORobotSurface2','StarlightCurves','SwedishLeaf','Symbols','SyntheticControl','Trace','TwoLeadECG','TwoPatterns','UWaveGestureLibraryX','UWaveGestureLibraryY','UWaveGestureLibraryZ','Wafer','WordSynonyms','Yoga']
redes = ['FCN','FCN_bilstm','FCN_bilstm_separated','bilstm_FCN','resnet','resnet_bilstm','resnet_bilstm_separated','bilstm_resnet','cnn','cnn_lstm','cnn_bilstm','cnn_lstm_separated','cnn_lstm_separated2','multi_cnn','multi_cnn_lstm','multi_cnn_bilstm','multi_cnn_bilstm2','lstm_cnn']

"""
Load data
"""
# Hyperparameters
batch_size = 64 #min(X_train.shape[0]/10, 16)
epochs = 1500
dimensions = 3 #set en 3 o 4, dependiendo de la red neuronal a aplicar,
# si la primera capa es conv1d aplicar 3, si es conv2d, aplicar 4
for j in range(len(datasets)):
	X, y = load_data(direc, datasets[j], dimensions)
	print(X.shape)
	print(y.shape)

	X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
	i = 1
	red = 'bilstm_resnet'

	standarize = True
	rescale = False
	normalize = False
	preprocesing = False
	# Procesamiento de los datos
	if standarize == True:
	  	X_train1, X_test1 = standardize_data(X_train, X_test)

	if rescale == True:
		X_train, X_test = rescale_data(X_train, X_test)

	if normalize == True:
		X_train, X_test = normalize_data(X_train, X_test)

	if preprocesing == True:
		X_train, X_test = preprocesing_data(X_train, X_test)

	#print("Despues de ....")
	#plot_dist.plot_variable_distributions(X_train)
	path = 'Resultados por dataset/'+datasets[j]+'/'+red+'/'+datasets[j]+'_'+str(i)
	#n_timesteps, n_features, n_outputs = X_train.shape[1], X_train.shape[2], y_train.shape[1]
	model, history = m_bilstm_resnet.bilstm_resnet(X_train, y_train, X_test, y_test, epochs, batch_size, path, earlystop = 1)

	# plot del entrenamiento y test
	plot_history(history, path)

	path2 = 'Resultados por dataset/'+datasets[j]+'/'+red+'/'+datasets[j]+'_'+str(i)+'_2500_epochs_'+str(batch_size)+'_batch'

	save_hist(path2+'_hist', history)
	#info.save_model(path2+'_model', model)
	saved_model = load_model(path+'_best_model.h5')

	# Evaluación del modeloX
	_, score = saved_model.evaluate(X_test, y_test, batch_size=batch_size, verbose=2)
	#_, score = model.evaluate([X_test, X_test], y_test, batch_size=batch_size, verbose=2)
	#_, score = model.evaluate([X_test, X_test,X_test], y_test, batch_size=batch_size, verbose=2)

	print(datasets[j])
	score = score * 100.0
	print('score: %.3f' % (score))

	y_score = saved_model.predict(X_test)
	#y_score = model.predict([X_test,X_test])
	#y_score = model.predict([X_test, X_test, X_test])

	y_score_classes = y_score.argmax(axis=-1)
	y_test_c = y_test.argmax(axis=-1)
	n_classes = y_test.shape[1]

	### Curva roc y area bajo la curva

	fpr, tpr, roc_auc = compute_roc(y_test, y_score)

	resultados = dict()

	print('micro-average ROC curve (area = %8.2f)' % float(format(roc_auc["micro"])))

	print('macro-average ROC curve (area = %8.2f)' % float(format(roc_auc["macro"])))

	#roc_auc_macro.append(float(roc_auc["macro"]))
	#roc_auc_micro.append(float(roc_auc["micro"]))
	resultados["roc_auc_macro"] = float(roc_auc["macro"])
	resultados["roc_auc_micro"] = float(roc_auc["micro"])

	plot_curvaROC(fpr, tpr, roc_auc, path)


	#confusion_matrix = evaluate.get_confusion_matrix(y_test_c, y_score_classes)
	#resultados["confusion_matrix"] = confusion_matrix .tolist()

	resultados["kappa"] = get_kappa(y_test_c, y_score_classes)
	#kappa.append(kappa_c)

	#report = evaluate.get_classification_report(y_test_c, y_score_classes)

	resultados["log_loss"] = get_log_loss(y_test, y_score)
	#log_loss.append(log_loss_c)

	resultados["accuracy"] = get_accuracy(y_test_c, y_score_classes)
	#accuracy.append(accuracy_c)

	f1_macro_c, f1_micro_c = get_f1(y_test_c, y_score_classes)
	resultados["f1_macro"] = f1_macro_c
	resultados["f1_micro"] = f1_micro_c

	recall_macro_c, recall_micro_c = get_recall(y_test_c, y_score_classes)
	resultados["recall_macro"] = recall_macro_c
	resultados["recall_micro"] = recall_micro_c

	precision_macro_c, precision_micro_c = get_precision(y_test_c, y_score_classes)
	resultados["precision_macro"] = precision_macro_c
	resultados["precision_micro"] = precision_micro_c
	save_results('Resultados por dataset/'+datasets[j]+'/'+red+'/'+datasets[j]+'_resultados'+'_'+str(i), resultados)