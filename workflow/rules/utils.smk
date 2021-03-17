import os

def forecast_stacking_dirname(wildcards, output):
    path = str(rules.forecast_model.output[0])
    return os.path.split(os.path.split(path)[0])[0]
