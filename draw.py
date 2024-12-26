from tkinter import *
from PIL import Image, ImageGrab
import torch
import torchvision.transforms as transforms
from models import Model_1
from tokenizer import LaTeXTokenizer


device = 'cuda' if torch.cuda.is_available() else 'cpu'
tokenizer = LaTeXTokenizer()
tokenizer.id_to_token = {0: '[PAD]', 1: '[BOS]', 2: '[EOS]', 3: '[UNK]', 4: '{', 5: '}', 6: '_', 7: 'x', 8: '^', 9: ')', 10: '(', 11: '=', 12: '1', 13: 'i', 14: 't', 15: '2', 16: 'a', 17: 'r', 18: 'm', 19: '-', 20: '\\frac', 21: '0', 22: 'd', 23: 'n', 24: ',', 25: '+', 26: '\\\\', 27: '\\begin', 28: '\\end', 29: '|', 30: 'k', 31: '&', 32: 'f', 33: '\\int', 34: '\\sqrt', 35: '3', 36: 'y', 37: 'p', 38: '\\hat', 39: 'A', 40: 's', 41: ']', 42: '[', 43: '\\partial', 44: 'c', 45: 'e', 46: '\\tilde', 47: '.', 48: '/', 49: '4', 50: 'X', 51: 'b', 52: 'j', 53: 'P', 54: 'v', 55: 'C', 56: 'S', 57: 'g', 58: 'u', 59: 'R', 60: 'z', 61: 'T', 62: 'l', 63: '\\prime', 64: 'E', 65: 'N', 66: '\\overline', 67: 'F', 68: 'B', 69: 'L', 70: 'V', 71: '5', 72: 'o', 73: '\\mu', 74: 'q', 75: 'I', 76: '\\cdot', 77: 'M', 78: '\\alpha', 79: '\\pi', 80: 'H', 81: 'D', 82: '\\}', 83: '\\{', 84: '6', 85: 'h', 86: '\\in', 87: 'G', 88: '\\sum', 89: '\\lambda', 90: 'K', 91: '*', 92: '\\prod', 93: '<', 94: 'w', 95: '\\theta', 96: 'Q', 97: '\\sigma', 98: ':', 99: '\\infty', 100: 'U', 101: '\\omega', 102: 'Y', 103: '\\rho', 104: 'Z', 105: '\\rangle', 106: '\\beta', 107: '7', 108: '\\rightarrow', 109: '\\gamma', 110: '\\epsilon', 111: 'O', 112: '\\underline', 113: '\\phi', 114: '\\le', 115: '\\notin', 116: '\\varphi', 117: 'W', 118: '\\delta', 119: '\\psi', 120: '8', 121: '\\nu', 122: '>', 123: '\\vec', 124: '\\langle', 125: '\\Delta', 126: 'J', 127: '\\times', 128: '\\dot', 129: '\\Omega', 130: '!', 131: '\\tau', 132: '9', 133: '\\pm', 134: '\\chi', 135: '\\approx', 136: '\\eta', 137: ';', 138: '\\nabla', 139: '\\mathbb', 140: '\\xi', 141: '\\Phi', 142: '\\ge', 143: '\\Psi', 144: '\\Sigma', 145: '\\sim', 146: '\\zeta', 147: '\\circ', 148: '\\Gamma', 149: '\\ne', 150: '\\forall', 151: '\\Lambda', 152: '\\mapsto', 153: '\\otimes', 154: '\\hbar', 155: '\\cup', 156: '\\equiv', 157: '\\kappa', 158: '\\Pi', 159: '\\oplus', 160: '\\subset', 161: '\\cap', 162: '\\bigcup', 163: '\\subseteq', 164: '\\wedge', 165: '\\cong', 166: '\\neg', 167: '\\Theta', 168: '\\dagger', 169: '\\oint', 170: '\\Rightarrow', 171: '\\aleph', 172: '\\lfloor', 173: '\\rfloor', 174: '\\backslash', 175: '\\emptyset', 176: '\\perp', 177: '\\#', 178: '\\propto', 179: '\\%', 180: '\\simeq', 181: '\\vee', 182: '?', 183: '\\ll', 184: '\\Vdash', 185: '\\Xi', 186: '\\leftarrow', 187: '\\bigcap', 188: '\\longrightarrow', 189: '\\bullet', 190: '\\exists', 191: '\\iint', 192: '\\vdash', 193: '\\iff', 194: '\\top', 195: '\\|', 196: '\\bigoplus', 197: '\\odot', 198: '\\lceil', 199: '\\rceil', 200: '\\leftrightarrow', 201: '\\models', 202: '\\supseteq', 203: '\\bigwedge', 204: '\\varsigma', 205: '\\rightleftharpoons', 206: '\\angle', 207: '\\vdots', 208: '\\Leftrightarrow', 209: '\\subsetneq', 210: '\\iota', 211: '\\gg', 212: '\\ominus', 213: '\\supset', 214: '\\Upsilon', 215: '\\triangle', 216: '\\_'}
model = Model_1(vocab_size=217, d_model=256, nhead=8, dim_FF=1024, dropout=0, num_layers=3)
transform = transforms.Compose([transforms.Resize((512, 384)),transforms.ToTensor()])
model.load_state_dict(torch.load('runs/Exp8E8End_Acc=0.6702922582626343.pt', map_location = device, weights_only=True))
model.eval()
model.to(device)


def greedy(input):
    input = transform(input).unsqueeze(0).to(device) # -> [B, C, H, W]
    with torch.no_grad():
        tgt_in = torch.ones([1, 1], dtype=torch.long).to(device)
        tgt_mask = torch.triu(torch.ones(200, 200) * float("-inf"), diagonal=1).to(device)
        for i in range(1, 200):
            output = model(input, tgt_in, tgt_mask[:i, :i])
            next_token = torch.argmax(output, dim=-1, keepdim=True)
            tgt_in = torch.cat((tgt_in, next_token[:,-1]), dim=1)
            if int(next_token[:, -1]) == 2:
                 break

    latex_out = tokenizer.decode(t for t in tgt_in[0, :].tolist())
    print(latex_out)
    return latex_out


class Paint(object):

    DEFAULT_PEN_SIZE = 5.0
    DEFAULT_COLOR = 'black'

    def __init__(self):
        self.root = Tk()

        self.pen_button = Button(self.root, text='pen', command=self.use_pen)
        self.pen_button.grid(row=0, column=0)

        self.brush_button = Button(self.root, text='brush', command=self.use_brush)
        self.brush_button.grid(row=0, column=1)

        self.clear_button = Button(self.root, text='clear', command=self.clear)
        self.clear_button.grid(row=0, column=2)
    
        self.eraser_button = Button(self.root, text='eraser', command=self.use_eraser)
        self.eraser_button.grid(row=0, column=3)

        self.predict_button = Button(self.root, text="predict", command=self.predict)
        self.predict_button.grid(row=0, column=4)

        self.choose_size_button = Scale(self.root, from_=10, to=100, orient=HORIZONTAL)
        self.choose_size_button.grid(row=0, column=5)

        self.width, self.height = self.root.winfo_screenwidth(), self.root.winfo_screenheight()
        self.c = Canvas(self.root, bg='white', width=self.width, height=self.height)
        self.c.grid(row=1, columnspan=6)
        
        self.setup()
        self.root.mainloop()


    def setup(self):
        self.root.attributes('-fullscreen', True)
        self.old_x = None
        self.old_y = None
        self.line_width = self.choose_size_button.get()
        self.color = self.DEFAULT_COLOR
        self.eraser_on = False
        self.active_button = self.pen_button
        self.c.bind('<B1-Motion>', self.paint)
        self.c.bind('<ButtonRelease-1>', self.reset)
        self.root.bind("<Escape>", self.exit_fullscreen)

    def exit_fullscreen(self, event=None):
        self.root.attributes('-fullscreen', False)
        self.root.quit()

    def use_pen(self):
        self.activate_button(self.pen_button)

    def use_brush(self):
        self.activate_button(self.brush_button)

    def clear(self):
        self.c.delete("all")

    def use_eraser(self):
        self.activate_button(self.eraser_button, eraser_mode=True)

    def activate_button(self, some_button, eraser_mode=False):
        self.active_button.config(relief=RAISED)
        some_button.config(relief=SUNKEN)
        self.active_button = some_button
        self.eraser_on = eraser_mode

    def paint(self, event):
        self.line_width = self.choose_size_button.get()
        paint_color = 'white' if self.eraser_on else self.color
        if self.old_x and self.old_y:
            self.c.create_line(self.old_x, self.old_y, event.x, event.y,
                               width=self.line_width, fill=paint_color,
                               capstyle=ROUND, smooth=TRUE, splinesteps=36)
        self.old_x = event.x
        self.old_y = event.y

    def reset(self, event):
        self.old_x, self.old_y = None, None


    def predict(self):
        screen = ImageGrab.grab(bbox=(0, 75, self.width*1.5, self.height*1.5))
        screen.show()
        latex_out = greedy(screen)


if __name__ == '__main__':
    Paint() 