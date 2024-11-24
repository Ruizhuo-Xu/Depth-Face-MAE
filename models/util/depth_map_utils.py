import torch
import torch.nn.functional as F

def gradient(x):
    # tf.image.image_gradients(image)
    h_x = x.size()[-2]
    w_x = x.size()[-1]
    # gradient step=1
    # l = x
    l = F.pad(x, [1, 0, 0, 0])[:, :, :, :-1]
    r = F.pad(x, [0, 1, 0, 0])[:, :, :, 1:]
    # t = x
    t = F.pad(x, [0, 0, 1, 0])[:, :, :-1, :]
    b = F.pad(x, [0, 0, 0, 1])[:, :, 1:, :]

    # dx, dy = torch.abs(r - l), torch.abs(b - t)
    dx, dy = (r - l), (b - t)
    # dx will always have zeros in the last column, r-l
    # dy will always have zeros in the last row,    b-t

    dx[:, :, :, 0] = 0
    dy[:, :, 0, :] = 0
    dx[:, :, :, -1] = 0
    dy[:, :, -1, :] = 0

    return dx, dy
    

def batch_calc_normal_map(img, sub, div):
    img_ = (img * div + sub) * 255.0
    zx, zy = gradient(img_)
    # normal = torch.concat([-zx, -zy, torch.ones_like(zx)], dim=1)
    # z y x逆序
    normal = torch.concat([torch.ones_like(zx), -zy, -zx], dim=1)
    # -1, 1
    normal = torch.nn.functional.normalize(normal, p=2, dim=1)
    # normal = (normal + 1) / 2 * 255
    return normal