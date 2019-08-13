import matplotlib.pyplot as plt
import pandas as pd

"""
Plot history of Model

Parameters:
    - history -> Model learning history
    - path -> Where it will be stored
"""

def plot_history(history, path):
    # list all data in history
    print(history.history.keys())
    # summarize history for accuracy
    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.savefig(path+'_hist_acc.png')
    plt.show()
    
    # summarize history for loss
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.savefig(path+'_hist_loss.png')
    plt.show()

    log = pd.DataFrame(history.history)
    #print(log)