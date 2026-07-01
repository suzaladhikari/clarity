import pickle
## Saving the model 
def saving_pickle(path, object):
    with open(path, 'wb') as f: 
        pickle.dump(object, f)

## Loading the model 

## Loading the pickle file 

def loading_pickle(path):
    with open(path, 'rb') as f:
        obj = pickle.load(f)
    return obj 