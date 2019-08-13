import json, codecs
import numpy as np
from keras.models import model_from_yaml

"""
Save history

Parameters:
    - path -> Where it will be stored
    - history -> model history
"""

def save_hist(path,history):
    new_hist = {}
    for key in list(history.history.keys()):
        if type(history.history[key]) == np.ndarray:
            new_hist[key] == history.history[key].tolist()
        elif type(history.history[key]) == list:
            if  type(history.history[key][0]) == np.float64:
                new_hist[key] = list(map(float, history.history[key]))

    #print(new_hist)
    with codecs.open(path, 'w', encoding='utf-8') as f:
        json.dump(new_hist, f, separators=(',', ':'), sort_keys=True, indent=4) 

    print("Saved history to disk")


"""
Load history

Parameters:
    - path -> DIrectory history
Output:
	- n -> model history
"""

def load_hist(path):
    with codecs.open(path, 'r', encoding='utf-8') as f:
        n = json.loads(f.read())

    print("Loaded history from disk")
    return n


"""
Save model

Parameters:
    - path -> Where it will be stored
    - model -> model to save
"""

def save_model(path, model):
    # serialize model to YAML
    model_yaml = model.to_yaml()
    with open(path+".yaml", "w") as yaml_file:
        yaml_file.write(model_yaml)
    # serialize weights to HDF5
    model.save_weights(path+".h5")
    print("Saved model to disk")


"""
Load model

Parameters:
    - path -> DIrectory history
Output:
	- loaded_model -> model
"""

def load_model(path):
    # load YAML and create model
    yaml_file = open(path+'.yaml', 'r')
    loaded_model_yaml = yaml_file.read()
    yaml_file.close()
    loaded_model = model_from_yaml(loaded_model_yaml)
    # load weights into new model
    loaded_model.load_weights(path+".h5")
    print("Loaded model from disk")
    return loaded_model


"""
Save history

Parameters:
    - path -> Where it will be stored
    - results -> results to save
"""

def save_results(path, results):
    with codecs.open(path, 'w', encoding='utf-8') as f:
        json.dump(results, f, separators=(',', ':'), indent=4) 