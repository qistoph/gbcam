import numpy as np
from PIL import Image

def load_sprite(filename):
    im = Image.open(filename)
    data = np.array(im)
    return data

def load_animated_sprite(filename, size):
    im = Image.open(filename)
    data = np.array(im)

    assert data.shape[0] == size[0], "Animated sprite height must equal frame height"
    assert data.shape[1] % size[1] == 0, "Animated sprite width must be multiple of frame width"

    frames = []
    for x in range(0, data.shape[1], size[1]):
        frame = data[:, x:x+size[1]]
        frames.append(frame)
    return frames

logo = load_sprite('nintendo.gif')

mario = load_sprite('mario.gif')

mario_walking = load_animated_sprite('mario_walking.gif', (30, 30))
