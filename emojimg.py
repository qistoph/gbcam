import emoji
import numpy as np
from PIL import Image, ImageFont, ImageDraw

font = ImageFont.truetype('/home/chris/Downloads/AppleColorEmoji.ttf', 109, encoding='unic')
textToDraw = emoji.emojize(':thumbs_up:')
(_,_,w,h) = font.getbbox(textToDraw)

im = Image.new('RGBA',(w,h),(0,0,0,0))
draw = ImageDraw.Draw(im)
draw.text((0,0), textToDraw, (0,0,0), embedded_color=True, font=font)

thumbs_up = np.array(im)
