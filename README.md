# HME_Training

Training code for handwritten mathematical expression recognition using a bidirectionally-trained transformer, and a canvas GUI to write math, perform inference, and copy resulting LaTeX to clipboard. See a quick demo here: https://youtu.be/TkWYrTANouE?feature=shared.


# Background

This project evolved from my [SPIS](https://spis.ucsd.edu/) final [project](https://github.com/catsandsoup32/Dynamic-Digit-Recognition) a few months prior, which was my first real coding experience apart from AP CSP. An ongoing attempt to deploy as a web app can be found [here](https://github.com/catsandsoup32/TeXan).


# Color Channel Embedding

Just as in text-to-text translation or image-captioning tasks, an encoder-decoder architecture is used here. The encoder is a pretrained Densenet-121 CNN and the decoder is a standard implementation of the one presented in *Attention is All You Need*. The pretrained Densenet means that all three color channels are inputted, and thus it is a bit of a waste to only process black-on-white images. For that reason, it becomes useful to embed time and distance information into color channels. This is performed with online InkML-format data (*online* as in opposed to *offline* image data) that represents the coordinates and timestamp of points sampled along pen strokes. For each line segment, its red, green, and blue channels are calculated by `timestamp / time of entire stroke`, `x-displacement * scaling_x / (max x - min x) `, and `y-displacement * scaling_y / (max y - min y)` respectively. The distance values were initially consistently less than 0.004 = 1/255, so the scaling was added to ensure their impact. Below are some examples of images modified in this way (and also with an applied erode/dilate transform).

<p align="center">
  <img src="public/color_ex.png" alt="Color Example" width="750">
</p>


# Bidirectionality

A single decoder is trained on both left-to-right and right-to-left sequences, and during inference, after beam search in each direction, each beam is compared to every beam in the reverse direction and its probability is adjusted accordingly.

LaTeX and math itself are inherently somewhat palindromic (`\begin{matrix}` is always followed by `\end{matrix}`, every opening bracket or parenthesis is always followed by a closing bracket or parenthesis, operands are always sandwiched between expressions, etc.), much more so than the English language, which reflects in this simpler approach compared to the more complex and deeper bidirectionality within some language models such as BERT.


# References

- The model architecture and bidirectionality are from [this paper](https://arxiv.org/abs/2105.02412) by Zhao et al. (2021)
- The idea to embed time and distance information into color channels came from [this paper](https://arxiv.org/html/2402.15307v1) by Fadeeva et al. (2024)
- Training samples are obtained from [MathWriting](https://arxiv.org/html/2404.10690v1) coming out of Google Research
- Timothy Leung's [blog](https://actamachina.com/) post was a great help in getting started with the PyTorch code


