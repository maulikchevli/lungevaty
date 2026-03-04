from monai import transforms
import vital.defined_transformations as vitalforms
import torch

def get_dtype(str_dtype):
    if str_dtype == "bfloat16":
        return torch.bfloat16

def make_transformations(tf_dict):
    transforms_list = []
    for tf, kwargs in tf_dict.items():
        is_user_defined = tf.split("_")[-1]
        tf = tf.split("_")[0]

        lib = vitalforms if is_user_defined == "our" else transforms
        if kwargs is not None:
            if 'dtype' in kwargs:
                new_kwargs = dict(kwargs)
                new_kwargs['dtype'] = get_dtype(kwargs['dtype'])
                new_tf = getattr(lib, tf)(**new_kwargs)
            else:
                new_tf = getattr(lib, tf)(**kwargs)
        else:
            new_tf = getattr(lib, tf)
        transforms_list.append(new_tf)

    _transforms = transforms.Compose(transforms_list)
    return _transforms