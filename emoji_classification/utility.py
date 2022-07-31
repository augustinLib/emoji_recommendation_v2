from sklearn.tree import export_text
import torch


# gradient간의 거리 측정
def get_grad_norm(parameters, norm_type = 2):
    parameters = list(filter(lambda p : p is not None, parameters))

    total_norm = 0

    try:
        for p in parameters:
            param_norms = p.grad.data.norm(norm_type)
            total_norm += param_norms ** norm_type
        total_norm = total_norm ** (1./norm_type)

    except Exception as e:
        print(e)

    return total_norm

# parameter간의 거리 측정
def get_parameters_norm(parameters, norm_type = 2):
    total_norm = 0

    try:
        for p in parameters:
            param_norm = p.data.norm(norm_type)
            total_norm += param_norm**norm_type

        total_norm = total_norm ** (1. / norm_type)

    except Exception as e:
        print(e)

    return total_norm