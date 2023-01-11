import numpy as np


def RSE(pred, true):
    return np.sqrt(np.sum((true - pred) ** 2)) / np.sqrt(np.sum((true - true.mean()) ** 2))


def CORR(pred, true):
    u = ((true - true.mean(0)) * (pred - pred.mean(0))).sum(0)
    d = np.sqrt(((true - true.mean(0)) ** 2 * (pred - pred.mean(0)) ** 2).sum(0))
    return (u / d).mean(-1)


def MAE(pred, true):
    return np.mean(np.abs(pred - true))


def MSE(pred, true):
    return np.mean((pred - true) ** 2)

def per_MAE(pred, true):
    per_MAE = np.zeros(len(pred[0][0]))
    for i in range(len(per_MAE)):
        per_MAE[i] = np.mean(np.abs(pred[:,:,i] - true[:,:,i] ) )
    return per_MAE

def per_MSE(pred, true):
    per_MSE = np.zeros(len(pred[0][0]))
    for i in range(len(per_MSE)):
        per_MSE[i] = np.mean((pred[:,:,i] - true[:,:,i]) ** 2)
    return per_MSE


def RMSE(pred, true):
    return np.sqrt(MSE(pred, true))


def MAPE(pred, true):
    return np.mean(np.abs((pred - true) / true))


def MSPE(pred, true):
    return np.mean(np.square((pred - true) / true))


def metric(pred, true):
    mae = MAE(pred, true)
    mse = MSE(pred, true)
    rmse = RMSE(pred, true)
    mape = MAPE(pred, true)
    mspe = MSPE(pred, true)
    per_mae = per_MAE(pred, true)
    per_mse = per_MSE(pred, true)

    return mae, mse, rmse, mape, mspe, per_mae, per_mse
