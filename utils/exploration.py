import numpy as np


def create_epsilon_func(mode, **kwargs):
    assert mode in ['cyclic'], "Unknown epsilon mode."

    if mode == "cyclic":
        return create_cycle_decay_fn(**kwargs)
    return None


def create_cycle_decay_fn(initial_value, final_value, cycle_len, num_cycles):
    max_step = cycle_len * num_cycles

    def eps_func(step):
        relative = 1. - step / max_step
        relative_cosine = 0.5 * (np.cos(np.pi * np.mod(step, cycle_len) / cycle_len) + 1.0)
        return relative_cosine * (initial_value - final_value) * relative + final_value

    return eps_func