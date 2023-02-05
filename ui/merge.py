import re
import numpy as np
from .safetensors import load_st
from .ckpt import load_torch

checkpoint_dict_skip_on_merge = ["cond_stage_model.transformer.text_model.embeddings.position_ids"]

def load_model(path):
    r = {}
    if path.endswith("ckpt"):
        r = load_torch(path)
    elif path.endswith("safetensors"):
        r = load_st(path)
    else:
        return None
    return r.get("state_dict", r)

def to_half(tensor, enable):
    if enable and tensor.dtype == np.float:
        return np.half(tensor)

    return tensor

def merge_weight(theta_0, theta_1, theta_2, interp_method, multiplier, save_as_half=False, vae_dict="", discard_weights=""):

    def weighted_sum(theta0, theta1, alpha):
        return ((1 - alpha) * theta0) + (alpha * theta1)

    def get_difference(theta1, theta2):
        return theta1 - theta2

    def add_difference(theta0, theta1_2_diff, alpha):
        return theta0 + (alpha * theta1_2_diff)

    theta_funcs = {
        "Weighted sum": (None, weighted_sum),
        "Add difference": (get_difference, add_difference),
        "No interpolation": (None, None),
    }
    theta_func1, theta_func2 = theta_funcs[interp_method]


    result_is_inpainting_model = False

        for key in theta_1.keys():
            if key in checkpoint_dict_skip_on_merge:
                continue

            if 'model' in key:
                if key in theta_2:
                    t2 = theta_2.get(key, np.zeros_like(theta_1[key]))
                    theta_1[key] = theta_func1(theta_1[key], t2)
                else:
                    theta_1[key] = np.zeros_like(theta_1[key])

        del theta_2

    print("Merging...")
    for key in theta_0.keys():
        if theta_1 and 'model' in key and key in theta_1:

            if key in checkpoint_dict_skip_on_merge:
                continue

            a = theta_0[key]
            b = theta_1[key]

            # this enables merging an inpainting model (A) with another one (B);
            # where normal model would have 4 channels, for latenst space, inpainting model would
            # have another 4 channels for unmasked picture's latent space, plus one channel for mask, for a total of 9
            if a.shape != b.shape and a.shape[0:1] + a.shape[2:] == b.shape[0:1] + b.shape[2:]:
                if a.shape[1] == 4 and b.shape[1] == 9:
                    raise RuntimeError("When merging inpainting model with a normal one, A must be the inpainting model.")

                assert a.shape[1] == 9 and b.shape[1] == 4, f"Bad dimensions for merged layer {key}: A={a.shape}, B={b.shape}"

                theta_0[key][:, 0:4, :, :] = theta_func2(a[:, 0:4, :, :], b, multiplier) #wtf
                result_is_inpainting_model = True
            else:
                theta_0[key] = theta_func2(a, b, multiplier)

            theta_0[key] = to_half(theta_0[key], save_as_half)


    del theta_1

    if vae_dict is not None:
        print(f"Baking in VAE")
        for key in vae_dict.keys():
            theta_0_key = 'first_stage_model.' + key
            if theta_0_key in theta_0:
                theta_0[theta_0_key] = to_half(vae_dict[key], save_as_half)

        del vae_dict

    if save_as_half and not theta_func2:
        for key in theta_0.keys():
            theta_0[key] = to_half(theta_0[key], save_as_half)

    if discard_weights:
        regex = re.compile(discard_weights)
        for key in list(theta_0):
            if re.search(regex, key):
                theta_0.pop(key, None)

    #fuck just throw this to convert.py
    return theta_0
