#!/usr/bin/python3

import time
import cv2
import numpy as np
from gbc_palettes import palettes
import pyfakewebcam
import PySimpleGUI as sg
import oyaml as yaml

config_filename = 'config.yaml'

try:
    with open(config_filename) as f:
        config = yaml.safe_load(f)
except FileNotFoundError:
    config = {}

if 'palette' not in config:
    config['palette'] = 'CRTGB'
if 'brightness' not in config:
    config['brightness'] = 0
if 'contrast'  not in config:
    config['contrast'] = 1
if 'gamma'  not in config:
    config['gamma'] = 1
if 'dither'  not in config:
    config['dither'] = 0.5

#OUT_SIZE=(1066, 600)
#OUT_SIZE=(800, 600)
OUT_SIZE=(1280, 720)

vid = cv2.VideoCapture(0)
fake = pyfakewebcam.FakeWebcam('/dev/video2', *OUT_SIZE)

# CV_CAP_PROP_FRAME_COUNT
#print("Frame default resolution: (" + str(vid.get(cv2.CAP_PROP_FRAME_WIDTH)) + "; " + str(vid.get(cv2.CAP_PROP_FRAME_HEIGHT)) + ")")
vid.set(cv2.CAP_PROP_FRAME_WIDTH, 800)
vid.set(cv2.CAP_PROP_FRAME_HEIGHT, 600)
print("Frame resolution set to: (" + str(vid.get(cv2.CAP_PROP_FRAME_WIDTH)) + "; " + str(vid.get(cv2.CAP_PROP_FRAME_HEIGHT)) + ")")

#print("Frame default count: " + str(vid.get(cv2.CAP_PROP_FRAME_COUNT)))

def resize(im, W=160, H=144):
    scale = W / im.shape[1]
    if scale * im.shape[0] < H:
        scale = H / im.shape[0]

    x_off = int((im.shape[1] - W/scale) // 2)
    y_off = int((im.shape[0] - H/scale) // 2)
    #box = (x_off, y_off, W/scale + x_off, H/scale + y_off)

    #print(scale, [y_off,y_off+int(H/scale), x_off,int(x_off+W/scale)])
    im_crop = im[y_off:y_off+int(H/scale), x_off:int(x_off+W/scale)]

    #return im.resize((W, H), box=box)
    #return cv2.resize(im_crop, dsize=(W, H), fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR)
    return cv2.resize(im_crop, dsize=(W, H), fx=scale, fy=scale, interpolation=cv2.INTER_NEAREST)

def greyscale(data, contrast=1.0, gamma=1.0):
    grey_factors = np.array([0.3, 0.59, 0.11]) / 255 # RGB ratios and bytes to units

    # RGB bytes to greyscale units
    data_grey = data.dot(grey_factors)
    # Apply contrast
    data_grey = (data_grey - 0.5) * contrast + 0.5
    # Limit darker than black or lighter than white
    data_grey = data_grey.clip(0, 1)
    # Apply gamma, no clip needed (x^gamma = 0->1, for x = 0->1)
    data_grey = np.power(data_grey, gamma)

    # Units to bytes
    data_grey = data_grey * 255
    return data_grey.astype(np.uint8)

def brightness(data, brightness):
    #data += int(brightness)
    data = np.where((255 - data) < brightness, 255, data+brightness)
    return data

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
    palette = np.array(palette, dtype=np.uint8)[:,[2,1,0]]
    return palette[data]

bayer8 = np.array(bayer_matrix(3))

layout = [
    [sg.Text("OpenCV Demo", size=(60, 1), justification="center")],
    [sg.Image(filename="", key="-IMAGE-")],
    [
        sg.Combo(list(filter(lambda d: d[0] != '_', palettes)), default_value=config['palette'], size=(20, 1), key="-PALETTE-")
    ],
    [
        sg.Text("Brightness"),
        sg.Slider(
            (-255, 255),
            config['brightness'],
            1,
            orientation="h",
            size=(40, 15),
            key="-BRIGTHNESS-",
            )
    ],
    [
        sg.Text("Contrast"),
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
        sg.Text("Gamma"),
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
        sg.Text("Dither"),
        sg.Slider(
            (0, 10),
            config['dither'],
            0.1,
            orientation="h",
            size=(40, 15),
            key="-DITHER-",
            )
    ],
]

window = sg.Window("GBCam", layout)

background = resize(cv2.imread("background2.png"), *OUT_SIZE)

while(True):
    event, values = window.read(timeout=20)
    if event == sg.WIN_CLOSED:
        break
    #print(values)

    ret, frame = vid.read()
    #print(frame.shape)
    #print(type(frame))

    palette = palettes['CRTGB']
    if values['-PALETTE-'] in palettes:
        palette = palettes[values["-PALETTE-"]]

    frame = resize(frame)
    frame = greyscale(frame, 2**values["-CONTRAST-"], 2**values["-GAMMA-"])
    frame = brightness(frame, values["-BRIGTHNESS-"])
    frame = bayerFilter(frame, values['-DITHER-'])
    frame = colorize(frame, palette)
    frame = resize(frame, 800, 720)

    xoff = (OUT_SIZE[0]-800)//2
    if xoff > 0:
        frame_bg = np.copy(background)
        frame_bg[:,xoff:-xoff,:] = frame
        frame = frame_bg

    imgbytes = cv2.imencode(".png", cv2.resize(frame, (400, 225)))[1].tobytes()
    window["-IMAGE-"].update(data=imgbytes)

    fake.schedule_frame(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

    config['palette'] = values['-PALETTE-']
    config['contrast'] = values['-CONTRAST-']
    config['brightness'] = values['-BRIGTHNESS-']
    config['dither'] = values['-DITHER-']

    wk = cv2.waitKey(1) & 0xFF

    if wk == ord('q'):
        break

    time.sleep(1/30.0)

with open(config_filename, 'w') as f:
    f.write(yaml.dump(config))

window.close()
vid.release()
cv2.destroyAllWindows()
