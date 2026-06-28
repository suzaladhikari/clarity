import xgboost as xgb


def build_xgboost_model(params = None):
    default_params = {
        'n_estimators': 200, 
        'max_depth': 6, 
        'learning_rate':0.05,
        'subsample':0.8, ## prevents overfitting
        'colsample_bytree':0.8, ## feature subsampling 
        'objective':'reg:squarederror', ## used for regression 
        'randomstate':42, ## Setting up the random seed
        'n_jobs': -1  ## Using all the cores
    }
    if params:
        default_params.update(params)
    
    return xgb.XGBRegressor(**default_params)
