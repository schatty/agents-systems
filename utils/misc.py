import os
from glob import glob
import imageio
import yaml
import queue


def make_gif(source_dir, output):
    """
    Make gif file from set of .jpeg images.
    Args:
        source_dir (str): path with .jpeg images
        output (str): path to the output .gif file
    Returns: None
    """
    batch_sort = lambda s: int(s[s.rfind('/')+1:s.rfind('.')])
    image_paths = sorted(glob(os.path.join(source_dir, "*.png")),
                         key=batch_sort)

    images = []
    for filename in image_paths:
        images.append(imageio.imread(filename))
    imageio.mimsave(output, images)


def read_config(path):
    """
    Return python dict from .yml file.
    Args:
        path (str): path to the .yml config.
    Returns (dict): configuration object.
    """
    with open(path, 'r') as ymlfile:
        cfg = yaml.load(ymlfile)

    return cfg


def empty_torch_queue(q):
    while True:
        try:
            t = q.get_nowait()
            del t
        except queue.Empty:
            break