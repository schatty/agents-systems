import os
from glob import glob
from PIL import Image


def make_gif(source_dir, output, ext='png'):
    """
    Make gif file from set of .jpeg images.
    Args:
        source_dir (str): path with .jpeg images
        output (str): path to the output .gif file
        ext (str): extension
    Returns: None
    """
    sort_n = lambda s: int(s[s.rfind('/')+1:s.find('.')])
    image_paths = sorted(glob(os.path.join(source_dir, f"*.{ext}")), key=sort_n)
    frames = []
    for path in image_paths:
        img = Image.open(path)
        frames.append(img)
    frames[0].save(output, format='GIF', append_images=frames[1:],
                   save_all=True, duration=80, loop=0)