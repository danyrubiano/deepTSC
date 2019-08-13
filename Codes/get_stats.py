import numpy as np
from sklearn.metrics import auc, roc_curve, classification_report, confusion_matrix, cohen_kappa_score, accuracy_score, log_loss, recall_score, precision_score, f1_score
from scipy import interp
import matplotlib.pyplot as plt

"""
The reported averages include micro average (averaging the total true positives, false negatives and false positives),
macro average (averaging the unweighted mean per label), weighted average (averaging the support-weighted mean per label) 
and sample average
"""

"""
Compute AUC ROC

Parameters:
    - y_test -> Time series classes test data
    - y_score -> Time series classes compute model
Ouputs:
    - fpr -> False positives rate
    - tpr -> True positives rate
    - roc_auc -> AUC ROC
"""

def compute_roc(y_test, y_score):
    # Compute ROC curve and ROC area for each class

    n_classes = y_test.shape[1]
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    
    for i in range(n_classes):
        #tpr[i], fpr[i], _ = np.testing.assert_warns(UndefinedMetricWarning, roc_curve, y_test[:,i], y_score[:,i])
        #assert_raises(ValueError, roc_auc_score, y_test, y_score)
        #fpr[i], tpr[i], _ = roc_curve(y_test[:, i], y_score[:, i])
        fpr[i], tpr[i], _ = roc_curve(y_test[:,i], y_score[:,i])
        roc_auc[i] = auc(fpr[i], tpr[i])
    
    # Compute micro-average ROC curve and ROC areafrom sklearn.metrics import auc, roc_curve
    fpr["micro"], tpr["micro"], _ = roc_curve(y_test.ravel(), y_score.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
    
    # Compute macro-average ROC curve and ROC area
    
    # First aggregate all false positive rates
    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))
    #print(all_fpr)

    # Then interpolate all ROC curves at this points
    mean_tpr = np.zeros_like(all_fpr)
    #print(mean_tpr)
    for i in range(n_classes):
        mean_tpr += interp(all_fpr, fpr[i], tpr[i])
        #print(mean_tpr[i])
    
    # Finally average it and compute AUC
    mean_tpr /= n_classes
    
    fpr["macro"] = all_fpr
    tpr["macro"] = mean_tpr
    roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

    return fpr, tpr, roc_auc
  

"""
Plot AUC ROC

Parameters:
    - fpr -> False positives rate
    - tpr -> True positives rate
    - roc_auc -> AUC ROC
"""

def plot_curvaROC(fpr, tpr, roc_auc, path):
    # Plot linewidth.
    lw = 2

    # Plot all ROC curves
    plt.figure(1)
    plt.plot(fpr["micro"], tpr["micro"],
            label='micro-average ROC curve (area = {0:0.2f})'
                  ''.format(roc_auc["micro"]),
            color='deeppink', linestyle=':', linewidth=4)
    
    plt.plot(fpr["macro"], tpr["macro"],
            label='macro-average ROC curve (area = {0:0.2f})'
                  ''.format(roc_auc["macro"]),
            color='navy', linestyle=':', linewidth=4)
    
    plt.plot([0, 1], [0, 1], 'k--', lw=lw)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Some extension of Receiver operating characteristic to multi-class')
    plt.legend(loc="lower right")
    plt.savefig(path+'_roc.png')
    plt.show()
    
    """
    # Zoom in view of the upper left corner.
    plt.figure(2)
    plt.xlim(0, 0.2)
    plt.ylim(0.8, 1)
    plt.plot(fpr["micro"], tpr["micro"],
            label='micro-average ROC curve (area = {0:0.2f})'
                  ''.format(roc_auc["micro"]),
            color='deeppink', linestyle=':', linewidth=4)
    
    plt.plot(fpr["macro"], tpr["macro"],
            label='macro-average ROC curve (area = {0:0.2f})'
                  ''.format(roc_auc["macro"]),
            color='navy', linestyle=':', linewidth=4)
    plt.plot([0, 1], [0, 1], 'k--', lw=lw)
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Some extension of Receiver operating characteristic to multi-class')
    plt.legend(loc="lower right")
    plt.show()
    """

"""
Get confussion matrix

Parameters:
    - y_test -> Time series classes test data
    - y_score -> Time series classes compute model
Ouputs:
    - matrix -> Confussion matrix
"""

def get_confusion_matrix(y_test, y_score):
    matrix = confusion_matrix(y_test, y_score)
    print(matrix)
    #tn, fp, fn, tp = confusion_matrix([0, 1, 0, 1], [1, 1, 1, 0]).ravel()
    return matrix

"""
Get classification report

Parameters:
    - y_test -> Time series classes test data
    - y_score -> Time series classes compute model
Ouputs:
    - report -> Classification report
"""

def get_classification_report(y_test, y_score):
    report = classification_report(y_test, y_score)
    """
    {'label 1': {'precision':0.5,
               'recall':1.0,
               'f1-score':0.67,
               'support':1},
    """
    print(report)
    return report

"""
Get Kappa Cohen coefficient

Parameters:
    - y_test -> Time series classes test data
    - y_score -> Time series classes compute model
Ouputs:
    - kappa -> Kappa Cohen coefficient
"""

def get_kappa(y_test, y_score):
    kappa = cohen_kappa_score(y_test, y_score)
    print(kappa)
    return kappa

"""
Get accuracy score

Parameters:
    - y_test -> Time series classes test data
    - y_score -> Time series classes compute model
Ouputs:
    - accuracy -> accuracy score
"""

def get_accuracy(y_test, y_score):
    accuracy = accuracy_score(y_test, y_score)
    print(accuracy)
    return accuracy

"""
Get logaritmical cross entropy loss

Parameters:
    - y_test -> Time series classes test data
    - y_score -> Time series classes compute model
Ouputs:
    - log_loss_c -> logaritmical cross entropy loss
"""

def get_log_loss(y_test, y_score):
    log_loss_c = log_loss(y_test, y_score)
    print(log_loss_c)
    return log_loss_c

"""
Get F1 score

Parameters:
    - y_test -> Time series classes test data
    - y_score -> Time series classes compute model
Ouputs:
    - f1_score_macro -> F1 score macro
    - f1_score_micro -> F1 score micro
"""

def get_f1(y_test, y_score):
    f1_score_macro = f1_score(y_test, y_score, average='macro')
    f1_score_micro = f1_score(y_test, y_score, average='micro')
    print(f1_score_macro, f1_score_micro)
    return f1_score_macro, f1_score_micro

"""
Get recall score

Parameters:
    - y_test -> Time series classes test data
    - y_score -> Time series classes compute model
Ouputs:
    - recall_score_macro -> recall score macro
    - recall_score_micro -> recall score micro
"""

def get_recall(y_test, y_score):
    recall_score_macro = recall_score(y_test, y_score, average='macro')
    recall_score_micro = recall_score(y_test, y_score, average='micro')
    print(recall_score_macro, recall_score_micro)
    return recall_score_macro, recall_score_micro

"""
Get precision score

Parameters:
    - y_test -> Time series classes test data
    - y_score -> Time series classes compute model
Ouputs:
    - precision_score_macro -> precision score macro
    - precision_score_micro -> precision score micro
"""

def get_precision(y_test, y_score):
    precision_score_macro = precision_score(y_test, y_score, average='macro')
    precision_score_micro = precision_score(y_test, y_score, average='micro')
    print(precision_score_macro, precision_score_micro)
    return precision_score_macro, precision_score_micro
