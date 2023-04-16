import pandas as pd

def Prediction_LGBM(trained_models:list=None, data:pd.DataFrame=None, feature_names:list=None):
    """
    mode: "predict", "predict_proba"
    """
    data_copy = data.copy()
    # if mode == "predict":
    #     for i, model in enumerate(trained_models):
    #         y_preds = model.predict(data_copy[feature_names])
    #         data_copy[f"model_{i}"] = y_preds
    #
    # else:
    for i, model in enumerate(trained_models):
        y_preds = model.predict(data_copy[feature_names])
        data_copy[f"model_{i}"] = y_preds
    
    return data_copy

if __name__ == '__main__':
    Prediction_LGBM()