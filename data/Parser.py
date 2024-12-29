import io
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
import xml.etree.ElementTree as ET
import numpy as np
import pandas as pd
import os
import math

from pathlib import Path
from PIL import Image


def parse_inkml(filename: str, ns = {'inkml': 'http://www.w3.org/2003/InkML'}):
    with open(filename, "r") as f:
        root = ET.fromstring(f.read())

    max_x, max_y, min_x, min_y = 1e-5, 1e-5, 1e5, 1e5
    strokes = []
    for trace in root.findall(".//inkml:trace", ns): # For all stroke ids
        stroke = []
        coords = trace.text.strip().split(',') # List as ['x y t', 'x1, y1, t1', . . .]
        for coord in coords:
            x, y, t = coord.split(' ')
            x, y, t = float(x), -float(y), float(t) # Note y is flipped
            if x > max_x: max_x = x
            if y > max_y: max_y = y
            if x < min_x: min_x = x
            if y < min_y: min_y = y
            stroke.append((x, y, t)) 
        if (len(stroke) == 1): # For cases where stroke is a dot
            stroke.append((stroke[0][0]+1, stroke[0][1]+1, stroke[0][2]+1))
        strokes.append(stroke) 

    x_scale, y_scale = abs(max_x - min_x), abs(max_y - min_y)
    latex = root.find('.//inkml:annotation[@type="normalizedLabel"]', ns).text.strip(" $") 
    splitTag = root.find('.//inkml:annotation[@type="splitTagOriginal"]', ns).text
    return strokes, latex, splitTag, x_scale, y_scale


# For black-on-white images - note: change this to fit with 2D array
def cache_data(data_dir, save_folder):
    fig, ax = plt.subplots()
    data_dir = Path(data_dir)
    for file in data_dir.glob("*/*.inkml"):
        strokes, latex, splitTag = parse_inkml(file)
        if splitTag != 'synthetic':
            img_file = os.path.join('data', save_folder, str(splitTag), str(file).removesuffix('.inkml')[-16:] + '.png') # Same tag as the inkml
            txt_file = os.path.join('data', save_folder, str(splitTag), str(file).removesuffix('.inkml')[-16:] + '.txt')
        else:
            img_file = os.path.join('data', save_folder, str('train'), str(file).removesuffix('.inkml')[-16:] + '.png') # Same tag as the inkml
            txt_file = os.path.join('data', save_folder, str('train'), str(file).removesuffix('.inkml')[-16:] + '.txt')

        # Render and save image
        ax.set_axis_off()
        ax.set_aspect("equal")
        x, y, t = [], [], []
        for itm in strokes:
            if itm == 'TE':
                x.append(np.nan) # Accounts for pen lift
                y.append(np.nan)
                t.append(np.nan)
            else:
                x.append(itm[0])
                y.append(itm[1])
                t.append(itm[2])
        ax.plot(x, y, color="black", linewidth=2)
        buf = io.BytesIO()
        plt.savefig(buf, bbox_inches="tight", pad_inches=0)
        plt.cla()
        buf.seek(0)
        img = Image.open(buf).convert("L")
        bgW, bgH = 512, 384
        background = Image.new("RGB", (bgW, bgH), (255, 255, 255))
        imW, imH = img.size
        offset = ((bgW - imW) // 2, (bgH - imH) // 2)
        background.paste(img, offset)
        background.save(img_file) # Pads and centers the images (avg. aspect ratio is 2.29 WH)

        # Save latex
        with open(txt_file, "w") as f: # w keyword overwrites
            f.write(latex)


# Caches images with time / x-directional / y-directional info embedded in RGB channels
def cache_extra(data_dir, save_folder):
    counter = 0
    data_dir = Path(data_dir)
    fig, ax = plt.subplots()
    data_dir = Path(data_dir)
    
    for file in data_dir.glob("*/*.inkml"):
        strokes, latex, splitTag, x_scale, y_scale = parse_inkml(file)
        if splitTag != 'synthetic':
            img_file = os.path.join('data', save_folder, str(splitTag), str(file).removesuffix('.inkml')[-16:] + '.png') # Same tag as the inkml
            txt_file = os.path.join('data', save_folder, str(splitTag), str(file).removesuffix('.inkml')[-16:] + '.txt')
        else:
            img_file = os.path.join('data', save_folder, str('train'), str(file).removesuffix('.inkml')[-16:] + '.png') # Same tag as the inkml
            txt_file = os.path.join('data', save_folder, str('train'), str(file).removesuffix('.inkml')[-16:] + '.txt')

        segments, colors = np.empty((0, 2, 2)), np.zeros((0, 3))
        for stroke in strokes:
            segment = np.empty((len(stroke)-1, 2, 2))
            color = np.zeros((len(stroke)-1, 3))

            max_dist_x, max_dist_y = 1e-5, 1e-5 # Ensures this can't be 0, divide-by-zero bad
            for idx, itm in enumerate(stroke):
                if idx == 0: continue
                x1, y1, t = itm[0], itm[1], (itm[2] - stroke[0][2])
                x0, y0 = stroke[idx-1][0], stroke[idx-1][1]
                segment[idx-1, 0, 0], segment[idx-1, 0, 1] = x0, y0
                segment[idx-1, 1, 0], segment[idx-1, 1, 1] = x1, y1

                # Needed to scale displacements as they were less than 0.004 = 1/255 
                dist_x, dist_y = abs(x1 - x0), abs(y1 - y0)
                if dist_x > max_dist_x: max_dist_x = dist_x
                if dist_y > max_dist_y: max_dist_y = dist_y

                color[idx-1, 0] = t / (stroke[-1][2] - stroke[0][2]) 
                color[idx-1, 1] = dist_x / x_scale
                color[idx-1, 2] = dist_y / y_scale

            color[:, 1] = color[:, 1] * (x_scale / max_dist_x / 2)
            color[:, 2] = color[:, 2] * (y_scale / max_dist_y / 2)
            segments = np.concatenate([segments, segment], axis=0)
            colors = np.concatenate([colors, color], axis=0)

        ax.set_axis_off()
        ax.set_aspect("equal")
        lc = LineCollection(segments, colors=colors, linewidth=2)
        ax.add_collection(lc)
        ax.autoscale()
        buf = io.BytesIO()
        plt.savefig(buf, bbox_inches="tight", pad_inches=0)
        plt.cla()
        buf.seek(0)
        img = Image.open(buf)
        bgW, bgH = 512, 384
        background = Image.new("RGB", (bgW, bgH), (255, 255, 255))
        imW, imH = img.size
        offset = ((bgW - imW) // 2, (bgH - imH) // 2)
        background.paste(img, offset)
        background.save(img_file) 
        print(counter)
        counter += 1


if __name__ == '__main__':        
    #cache_data('data/mathwriting_2024_excerpt', 'excerpt_cache')
    cache_extra('data/mathwriting_2024_excerpt', 'excerpt_cache')
    
