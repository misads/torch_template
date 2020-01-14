from .Model import Model

models = {
    'default': Model,

}


def get_model(model: str):
    if model in models:
        return models[model]
    else:
        return models['default']
