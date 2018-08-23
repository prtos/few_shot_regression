import torch


def pcc(x, y):
    x, y = x.view(-1), y.view(-1)
    mean_x = torch.mean(x)
    mean_y = torch.mean(y)
    xm = x - mean_x
    ym = y - mean_y
    r_num = torch.sum(xm * ym)
    r_den = torch.norm(xm, 2) * torch.norm(ym, 2)
    r = r_num / (r_den + 1e-8)
    r = max(min(r, 1.0), -1.0)
    return r


def r2(y_pred, y_true):
    y_pred, y_true = y_pred.view(-1), y_true.view(-1)
    mean_y_true = torch.mean(y_true)
    ss_tot = torch.sum(torch.pow(y_true.sub(mean_y_true), 2))
    ss_res = torch.sum(torch.pow(y_pred - y_true, 2))
    score = 1 - (ss_res/(ss_tot + 1e-8))
    return score


def vse(y_pred, y_true):
    return torch.std(torch.pow(y_pred - y_true, 2))


def mse(y_pred, y_true):
    return torch.mean(torch.pow(y_pred - y_true, 2))