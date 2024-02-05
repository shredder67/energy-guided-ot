def std_scale(data, mu, std):
    return (data - mu)/std


def std_scale_inv(data, mu, std):
    return data*std + mu
