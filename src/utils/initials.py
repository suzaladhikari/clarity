import pickle
## Saving the model 
def saving_pickle(path, object):
    with open(path, 'wb') as f: 
        pickle.dump(object, f)

## Loading the model 
