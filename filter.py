#!/usr/bin/env python3

# Using 
# - CV2 for webcam capture
# - v4l2loopback for 'webcam' output
# - numpy for filters
# to create a GameBoy camera like experience for webcam usage

import time
import datetime
import cv2
import numpy as np
from gbc_palettes import palettes
import pyfakewebcam
import FreeSimpleGUI as sg
import oyaml as yaml
from collections import namedtuple
import sprites
from PIL import Image

config_filename = 'config.yaml'

try:
    with open(config_filename) as f:
        config = yaml.safe_load(f)
except FileNotFoundError:
    config = {}

ConfigItem = namedtuple('ConfigItem', ['id','value_name','default_value'])

config_items = [
    ConfigItem('camera', '-CAMERA-', '/dev/video0'),
    ConfigItem('fakecam', '-FAKECAM-', '/dev/video2'),
    ConfigItem('mirror', '-MIRROR-', True),
    ConfigItem('background', '-BACKGROUND-', True),
    ConfigItem('mario', '-MARIO-', True),
    ConfigItem('palette', '-PALETTE-', 'CRTGB'),
    ConfigItem('brightness', '-BRIGHTNESS-', 0),
    ConfigItem('contrast', '-CONTRAST-', 1),
    ConfigItem('gamma', '-GAMMA-', 1),
    ConfigItem('dither', '-DITHER-', 0.5),
    ConfigItem('fps', '-FPS-', 10),
    ConfigItem('zoom', '-ZOOM-', 0),
]

for item in config_items:
    if item.id not in config:
        config[item.id] = item.default_value

config_changed = False
config_saveat = time.time()

next_frame_at = time.time()

def config_save():
    with open(config_filename, 'w') as f:
        f.write(yaml.dump(config))

#OUT_SIZE=(1066, 600)
#OUT_SIZE=(800, 600)
OUT_SIZE=(1280, 720)

CAP_SIZE=(1280, 720)

vid = cv2.VideoCapture(config['camera'])
fake = pyfakewebcam.FakeWebcam(config['fakecam'], *OUT_SIZE)

# CV_CAP_PROP_FRAME_COUNT
#print("Frame default resolution: (" + str(vid.get(cv2.CAP_PROP_FRAME_WIDTH)) + "; " + str(vid.get(cv2.CAP_PROP_FRAME_HEIGHT)) + ")")
vid.set(cv2.CAP_PROP_FRAME_WIDTH, CAP_SIZE[0])
vid.set(cv2.CAP_PROP_FRAME_HEIGHT, CAP_SIZE[1])
print("Frame resolution set to: (" + str(vid.get(cv2.CAP_PROP_FRAME_WIDTH)) + "; " + str(vid.get(cv2.CAP_PROP_FRAME_HEIGHT)) + ")")

#print("Frame default count: " + str(vid.get(cv2.CAP_PROP_FRAME_COUNT)))

def zoom(im, amount):
    ratio = im.shape[0] / im.shape[1]
    amount_x = int(amount)
    amount_y = int(amount * ratio)
    #print("zoom amount:", amount, "x:", amount_x, "y:", amount_y)
    res = im[amount_x:-amount_x-1, amount_y:-amount_y-1]
    #print(im.shape, "=>", res.shape)
    return res

def resize(im, W=160, H=144):
    scale = W / im.shape[1]
    #print("im.shape:", im.shape, "W,H:", (W,H), "scale:", scale)
    if scale * im.shape[0] < H:
        scale = H / im.shape[0]

    x_off = int((im.shape[1] - W//scale) // 2)
    y_off = int((im.shape[0] - H//scale) // 2)
    #box = (x_off, y_off, W/scale + x_off, H/scale + y_off)

    #print(scale, [y_off,y_off+int(H/scale), x_off,int(x_off+W/scale)])
    im_crop = im[y_off:y_off+int(H/scale), x_off:int(x_off+W/scale)]

    #return im.resize((W, H), box=box)
    #return cv2.resize(im_crop, dsize=(W, H), fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR)
    #print(im_crop.shape, (W, H), scale, scale)
    return cv2.resize(im_crop, dsize=(W, H), fx=scale, fy=scale, interpolation=cv2.INTER_NEAREST)

def greyscale(data, contrast=1.0, gamma=1.0, brightness=0.0):
    grey_factors = np.array([0.3, 0.59, 0.11]) / 255 # RGB ratios and bytes to units

    # RGB bytes to greyscale units
    data_grey = data.dot(grey_factors)
    # Apply contrast
    data_grey = (data_grey - 0.5) * contrast + 0.5 + brightness
    # Limit darker than black or lighter than white
    data_grey = data_grey.clip(0, 1)
    # Apply gamma, no clip needed (x^gamma = 0->1, for x = 0->1)
    data_grey = np.power(data_grey, gamma)

    # Units to bytes
    data_grey = data_grey * 255
    return data_grey.astype(np.uint8)

# https://gamedev.stackexchange.com/questions/130696/how-to-generate-bayer-matrix-of-arbitrary-size
def bit_reverse(x, n):
    return int(bin(x)[2:].zfill(n)[::-1], 2)

def bit_interleave(x, y, n):
    x = bin(x)[2:].zfill(n)
    y = bin(y)[2:].zfill(n)
    return int(''.join(''.join(i) for i in zip(y, x)), 2)

def bayer_entry(x, y, n):
    return bit_reverse(bit_interleave(x ^ y, y, n), 2*n)

def bayer_matrix(n):
    r = range(2**n)
    return [[bayer_entry(x, y, n) for x in r] for y in r]

def bayerFilter(data_grey, ditherFactor = 0.5):
    #print(bayer8)
    tiles = np.ceil(np.divide(data_grey.shape, bayer8.shape)).astype(int)
    bayer_f = np.tile(bayer8, tiles)[0:data_grey.shape[0],0:data_grey.shape[1]]

    data_B = data_grey + (bayer_f - 32) * ditherFactor
    data_B = (data_B / 64).clip(0, 3).round().astype(np.uint8)
    return data_B

def colorize(data, palette):
    #palette = np.array(palette, dtype=np.uint8)[:,[2,1,0]]
    palette = np.array(palette, dtype=np.uint8)
    return palette[data]

bayer8 = np.array(bayer_matrix(3))

window_default_title = "GBCam"

def title_update():
    title = window_default_title
    if config_changed:
        title += "*"
    window.TKroot.title(title)

slider_label_size = (10, 1)

layout = [
    [sg.Text("OpenCV Demo", size=(60, 1), justification="center")],
    [sg.Image(filename="", key="-IMAGE-")],
    [
        sg.Text("Palette"),
        sg.Combo(list(filter(lambda d: d[0] != '_', palettes)), default_value=config['palette'], size=(20, 1), key="-PALETTE-"),
        sg.Checkbox('Mirror preview', default=config['mirror'], key="-MIRROR-"),
        sg.Checkbox('Background', default=config['background'], key="-BACKGROUND-"),
        sg.Checkbox('Mario', default=config['mario'], key="-MARIO-"),
    ],
    [
        sg.Button("Logo"),
        sg.Button("Save"),
    ],
    [
        sg.Text("Brightness", size=slider_label_size),
        sg.Slider(
            (-1, 1),
            config['brightness'],
            0.1,
            orientation="h",
            size=(40, 15),
            key="-BRIGHTNESS-",
            )
    ],
    [
        sg.Text("Contrast", size=slider_label_size),
        sg.Slider(
            (-4, 4),
            config['contrast'],
            0.2,
            orientation="h",
            size=(40, 15),
            key="-CONTRAST-",
            )
    ],
    [
        sg.Text("Gamma", size=slider_label_size),
        sg.Slider(
            (-4, 4),
            config['gamma'],
            0.2,
            orientation="h",
            size=(40, 15),
            key="-GAMMA-",
            )
    ],
    [
        sg.Text("Dither", size=slider_label_size),
        sg.Slider(
            (0, 10),
            config['dither'],
            0.1,
            orientation="h",
            size=(40, 15),
            key="-DITHER-",
            )
    ],
    [
        sg.Text("FPS", size=slider_label_size),
        sg.Slider(
            (1, 30),
            config['fps'],
            1,
            orientation="h",
            size=(40, 15),
            key="-FPS-",
            )
    ],
    [
        sg.Text("Zoom", size=slider_label_size),
        sg.Slider(
            (0, 1),
            config['zoom'],
            .1,
            orientation="h",
            size=(40, 15),
            key="-ZOOM-",
            )
    ],
    [
        sg.Text("", key="-STATUS-"),
    ],
]

window = sg.Window(window_default_title, layout)

background = resize(cv2.imread("background2.png"), *OUT_SIZE)

def overlay_transparent(background_img, img_to_overlay_t, x, y, overlay_size=None):
    """
    @brief      Overlays a transparant PNG onto another image using CV2

    @param      background_img    The background image
    @param      img_to_overlay_t  The transparent image to overlay (has alpha channel)
    @param      x                 x location to place the top-left corner of our overlay
    @param      y                 y location to place the top-left corner of our overlay
    @param      overlay_size      The size to scale our overlay to (tuple), no scaling if None

    @return     Background image with overlay on top
    """

    bg_img = background_img.copy()

    if overlay_size is not None:
        img_to_overlay_t = cv2.resize(img_to_overlay_t.copy(), overlay_size)

    # Extract the alpha mask of the RGBA image, convert to RGB
    b,g,r,a = cv2.split(img_to_overlay_t)
    overlay_color = cv2.merge((b,g,r))

    # Apply some simple filtering to remove edge noise
    mask = cv2.medianBlur(a,5)

    h, w, _ = overlay_color.shape
    roi = bg_img[y:y+h, x:x+w]

    # Black-out the area behind the logo in our original ROI
    img1_bg = cv2.bitwise_and(roi.copy(),roi.copy(),mask = cv2.bitwise_not(mask))

    # Mask out the logo from the logo image.
    img2_fg = cv2.bitwise_and(overlay_color,overlay_color,mask = mask)

    # Update the original image with our new ROI
    bg_img[y:y+h, x:x+w] = cv2.add(img1_bg, img2_fg)

    return bg_img

def overlay_sprite(im, sp, y, x):
    #im[y:y+sp.shape[0], x:x+sp.shape[1]] = sp

    if x < 0:
        sp = sp[:, -x:]
        x = 0

    if y < 0:
        sp = sp[-y:, :]
        y = 0

    h = min(sp.shape[0], im.shape[0] - y)
    w = min(sp.shape[1], im.shape[1] - x)

    #print("x,y:", (x, y))
    #print("w,h:", (w, h))

    if h <= 0 or w <= 0:
        return im

    sp = sp[:h, :w]

    im[y:y+h, x:x+w] = np.where(sp < 4, sp, im[y:y+h, x:x+w])
    return im

class Animation:
    def __init__(self, sprite, pos=(0,0), speed=(0,0)):
        self.frames = sprite
        self.frame_idx = 0
        self.pos = np.array(pos)
        self.speed = np.array(speed)

    def update(self):
        self.pos += self.speed
        self.frame_idx = (self.frame_idx + 1) % len(self.frames)

    def overlay(self, frame):
        #print("overlay sprite at", self.pos)
        return overlay_sprite(frame, self.frames[self.frame_idx], *self.pos)

class Mario (Animation):
    def __init__(self):
        super().__init__([cv2.flip(f, 1) for f in sprites.mario_walking], (114,-30), (0, 4))

    def jump(self):
        if self.speed[0] == 0 and self.pos[0] == 114:
            self.frames = [cv2.flip(sprites.mario_jump, 1)]
            self.frame_idx = 0
            self.speed[0] = -5

    def update(self):
        super().update()

        if self.pos[0] > 114:
            self.pos[0] = 114
            self.speed[0] = 0
            self.frames = [cv2.flip(f, 1) for f in sprites.mario_walking]
        elif self.pos[0] < 114:
            self.speed[0] += 1

        #print("speed:", self.speed, "pos:", self.pos)

mario_walking = Mario()

def camera_image():
    ret, frame = vid.read()
    #print(frame.shape)
    #print(type(frame))

    if ret:
        frame = zoom(frame, values["-ZOOM-"]*(min(CAP_SIZE)//2-2))
        frame = resize(frame)
        frame = greyscale(frame, 2**values["-CONTRAST-"], 2**values["-GAMMA-"], values["-BRIGHTNESS-"])
        frame = bayerFilter(frame, values['-DITHER-'])
        #frame = overlay_sprite(frame, cv2.flip(sprite, 1), 114, mw_x)

        return frame
    return np.ones([144, 160], dtype=np.uint8)

logo_times = [4, 1] # motion, stationary
logo_done_at = 0

def logo_image():
    frame = 3 * np.ones((144, 160), dtype=np.uint8)

    logo_at_p = 1 - (logo_done_at - time.time()) / sum(logo_times)
    #print(f'{logo_at_p:.2}')
    logo_at_p = min(1.0, (sum(logo_times)/logo_times[0]) * logo_at_p)
    #print(int(logo_done_at - time.time()), "p:", logo_at_p)

    min_y = -sprites.logo.shape[0]
    max_y = (frame.shape[0] - sprites.logo.shape[0] ) // 2

    y = int((logo_at_p) * (max_y - min_y) + min_y)
    x = (frame.shape[1] - sprites.logo.shape[1]) // 2
    #print(min_y, max_y, y, f'{logo_at_p:.2}')
    #print((y,x))

    overlay_sprite(frame, sprites.logo, y, x)

    return frame

def save_frame(frame, postfix):
    filename = datetime.datetime.now().strftime('%Y%m%d_%H%M%S') + postfix
    window["-STATUS-"].update(f"Saving as {filename}")
    with open(filename, 'wb') as f:
        Image.fromarray(frame).save(f)
    window["-STATUS-"].update(f"{filename} saved")

def update_frame(save = False):
    if time.time() < logo_done_at:
        frame = logo_image()
    else:
        frame = camera_image()

    mario_walking.update()
    if mario_walking.pos[1] == 42:
        mario_walking.jump()
    if mario_walking.pos[1] >= 160:
        mario_walking.pos[1] -= 190 # screen (160) + sprite (30)

    if values['-MARIO-'] and frame is not None:
        frame = mario_walking.overlay(frame)

    gb_ratio_width = 10*OUT_SIZE[1]//9

    palette = palettes['CRTGB']
    if values['-PALETTE-'] in palettes:
        palette = palettes[values["-PALETTE-"]]

    frame = colorize(frame, palette)
    if save: save_frame(frame, '.png')
    frame = resize(frame, gb_ratio_width, OUT_SIZE[1]) # HD height, with 10:9 ratio

    xoff = (OUT_SIZE[0]-gb_ratio_width)//2
    if xoff > 0:
        if values['-BACKGROUND-']:
            frame_bg = np.copy(background)
        else:
            frame_bg = np.zeros(background.shape).astype(np.uint8)
        frame_bg[:,xoff:-xoff,:] = frame
        frame = frame_bg

    preview = frame
    if values['-MIRROR-']:
        preview = cv2.flip(preview, 1)

    preview_size = (225*frame.shape[1]//frame.shape[0], 225)
    #print(frame.shape, preview_size)
    imgbytes = cv2.imencode(".png", 
            cv2.resize(
                cv2.cvtColor(preview, cv2.COLOR_BGR2RGB),
                preview_size)
            )[1].tobytes()
    window["-IMAGE-"].update(data=imgbytes)

    #fake.schedule_frame(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    fake.schedule_frame(frame)

save_next_frame = False
while(True):
    event, values = window.read(timeout=20)
    if event == sg.WIN_CLOSED:
        break

    if event == 'Logo':
        logo_done_at = time.time() + sum(logo_times)

    if event == 'Save':
        save_next_frame = True

    if time.time() > next_frame_at:
        next_frame_at = time.time() + 1/config['fps']
        update_frame(save_next_frame)
        save_next_frame = False

    for item in config_items:
        if item.value_name in values:
            if config[item.id] != values[item.value_name]:
                config[item.id] = values[item.value_name]
                config_changed = True
                config_saveat = time.time() + 2
                title_update()

    if config_changed and config_saveat < time.time():
        config_save()
        config_changed = False
        title_update()

    wk = cv2.waitKey(1) & 0xFF

    if wk == ord('q'):
        break

config_save()
print("Config saved")
window.close()
print("Window closed")
vid.release()
print("Video released")
cv2.destroyAllWindows()
print("Windows destroyed")
