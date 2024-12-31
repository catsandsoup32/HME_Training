# HME_Training

Training code for handwritten mathematical expression recognition using a bidirectionally-trained transformer, and a canvas GUI to write math, perform inference, and copy resulting LaTeX to clipboard. See the model in action being used to write out the softmax derivative: TBD.

# Setup

Run locally in `draw.py` file after `pip install -r requirements.txt` and ensuring LaTeX is installed. To train the model, check out the MathWriting link below to download the data, and then run `parser.py` after setting up destination folders. There is also an ongoing attempt to deploy the model online: https://github.com/catsandsoup32/TeXan.

# Color Channel Embedding

Just as in text-to-text translation or image-captioning tasks, an encoder-decoder architecture is used here. The encoder is a pretrained Densenet-121 CNN and the decoder is a standard implementation of the one presented in *Attention is All You Need*. The pretrained Densenet means that all three color channels are inputted, and thus it is a bit of a waste to only process black-on-white images. For that reason, it becomes useful to embed time and distance information into color channels. This is performed with online InkML-format data (*online* as in opposed to *offline* image data) that represents the coordinates and timestamp of points sampled along pen strokes. For each line segment, its red, green, and blue channels are calculated by `timestamp / time of entire stroke`, `x-displacement * scaling_x / (max x - min x) `, and `y-displacement * scaling_y / (max y - min y)` respectively. The distance values were initially consistently less than 0.004 = 1/255, so the scaling was added to ensure their impact. Below are some examples of images modified in this way (and also with an applied erode/dilate transform).

<p align="center">
  <img src="public/color_ex.png" alt="Color Example" width="750">
</p>




# Bidirectionality






# References

- The model architecture and bidirectionality are from [this paper](https://arxiv.org/abs/2105.02412) by Zhao et al. (2021)
- The idea to embed time and distance information into color channels came from [this paper](https://arxiv.org/html/2402.15307v1) by Fadeeva et al. (2024)
- Training samples are obtained from [MathWriting](https://arxiv.org/html/2404.10690v1) coming out of Google Research
- Timothy Leung's [blog](https://actamachina.com/) post was a great help in getting started with the PyTorch code


