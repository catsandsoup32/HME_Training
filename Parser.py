import io
import matplotlib.pyplot as plt
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

    strokes = []
    for trace in root.findall(".//inkml:trace", ns): # For all stroke ids
        coords = trace.text.strip().split(',') # List as ['x y t', 'x1, y1, t1', . . .]
        for coord in coords:
            x, y, t = coord.split(' ')
            strokes.append((float(x), -float(y), float(t)))
        strokes.append('TE') # trace end

    latex = root.find('.//inkml:annotation[@type="normalizedLabel"]', ns).text.strip(" $") 
    splitTag = root.find('.//inkml:annotation[@type="splitTagOriginal"]', ns).text

    return strokes, latex, splitTag


def cache_data(data_dir, save_folder):
    fig, ax = plt.subplots()
    data_dir = Path(data_dir)
    for file in data_dir.glob("*/*.inkml"):
        strokes, latex, splitTag = parse_inkml(file)
        img_file = os.path.join('HME_Training', 'data', save_folder, str(splitTag), str(file).removesuffix('.inkml')[-16:] + '.png') # Same tag as the inkml
        txt_file = os.path.join('HME_Training', 'data', save_folder, str(splitTag), str(file).removesuffix('.inkml')[-16:] + '.txt')
        csv_file = os.path.join('HME_Training', 'data', save_folder, str(splitTag), str(file).removesuffix('.inkml')[-16:] + '.csv')

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

        # Save online strokes as *offset coords and pen lift flags
        minX, maxX = min(x), max(x)
        minY, maxY = min(y), max(y)
        minT, maxT = min(t), max(t)
        xRange, yRange, tRange = maxX - minX, maxY - minY, maxT - minT

        # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! Change this later? Make sure inference matches
        PL_idx_List = []
        for idx, _ in enumerate(x): # Normalizes coordinates  
            x[idx] = ( x[idx] - minX ) / (xRange) if not np.isnan(x[idx]) else '[PL]'
            y[idx] = ( y[idx] - minY ) / (yRange) if not np.isnan(y[idx]) else '[PL]'
            t[idx] = ( t[idx] - minT ) / (tRange) if not np.isnan(t[idx]) else '[PL]'
            if np.isnan(x[idx]):
                PL_idx_List.append(idx)

        for idx, _ in enumerate(x): # Separates into strokes
            pass 


        data = {'x': x, 'y': y, 't': t}
        df = pd.DataFrame(data)
        df.to_csv(csv_file, index=False)


if __name__ == '__main__':        
    cache_data('HME_Training/data/mathwriting_2024_excerpt', 'excerpt_cache')

# Run this once normalization is settled (abs coords can be used) 
# full_cache hasn't been populated