import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from tensorflow import keras
import os

OVERLAP = 2 #overlap, so that the AI has a bit of context when generating the tiles for an image (can't be to high, or individual tiles will be visible)

def get_tiles(img):
    tiles = []
    w, h = img.size
    for i, y in enumerate(range(0, h, 64 - OVERLAP)):
        tiles.append([])
        for x in range(0, w, 64 - OVERLAP):
            #if the tile doesn't fit, crop the ends
            if y + 64 >= h:
                endy = h
            else:
                endy = y + 64
            if x + 64 >= w:
                endx = w
            else:
                endx = x + 64
            tile = img.crop((x, y, endx, endy))
            tile = add_padding(tile)
            tiles[i].append(tile)
    return tiles

def upscale_tiles(tiles):
    upscaled_tiles = []
    for i in range(len(tiles)):
        upscaled_tiles.append([])
        for tile in tiles[i]:
            pred = Model.predict(np.array([keras.utils.img_to_array(tile) / 255.0]))
            upscaled_tiles[i].append(Image.fromarray(np.uint8(pred[0] * 255)))  
    return upscaled_tiles       

def connect_tiles(tiles_upscaled, w, h):
    img = Image.new("RGB", (w, h))
    for i1, y in enumerate(range(0, h, 128 - 2 * OVERLAP)):
        for i2, x in enumerate(range(0, w, 128 - 2 * OVERLAP)):
            #if the tile doesn't fit, crop the ends
            if y + 128 >= h:
                endy = h - y
            else:
                endy = 128
            if x + 128 >= w:
                endx = w - x
            else:
                endx = 128
            tile = tiles_upscaled[i1][i2].crop((0, 0, endx, endy))
            img.paste(tile, (x, y))
    return img

def add_padding(img, dims=(64, 64)):
    img_padded = Image.new("RGB", dims)
    img_padded.paste(img, (0, 0))
    return img_padded

if __name__ == "__main__":
    filedir = os.getcwd()
    img = Image.open("demo_img3.png").convert("RGB")
    w, h = img.size
    Model = keras.models.load_model(f"{filedir}/model/model_1.h5")

    tiles = get_tiles(img)
    tiles = upscale_tiles(tiles)
    output_img = connect_tiles(tiles, 2 * w, 2 * h)
    plt.imshow(output_img)
    plt.show()