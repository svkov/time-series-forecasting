import pandas as pd
import os


# def get_results(models, path_to_pred, ticker):
#     model_names = models.split()
#     model_results = [path_to_pred + model for model in model_names]
#
#     results = pd.DataFrame(columns=model_names)
#     for result, model in zip(model_results, model_names):
#         path = os.path.join(result, f'{ticker}.csv')
#         df = pd.read_csv(path, parse_dates=True, index_col=0)
#         series = pd.Series(df.values[:, -1], index=df.index)  # фильтровать по name
#         results[model] = series
#     return results

def get_results(df, models, n_pred):
    models = models.split()
    results = pd.DataFrame()
    for model in models:
        # for i in range(n_pred):
        results[model] = df[f'{model} Close n{n_pred}'].iloc[n_pred:]
    return results