from tkinter import Tk, font, Canvas, StringVar, Label, Button, Scale, HORIZONTAL, RAISED, SUNKEN, ROUND, TRUE, Frame # dont use wildcard import
from tkinter.colorchooser import askcolor
import threading

from PIL import Image, ImageGrab, ImageTk
import PIL.ImageOps    
import pyautogui
import pygetwindow as gw
import io 
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure

import mss
import cv2

import sympy
from sympy import *

import torch
import torch.nn as nn
import torchvision.transforms.functional as TF
import torchvision.transforms as transforms

from Models import Model_1

transform_define = transforms.Compose([
    transforms.ToTensor()
])

class Paint(object):

    DEFAULT_PEN_SIZE = 5.0
    DEFAULT_COLOR = 'black'

    def __init__(self, model, model_folder, transform):
        self.root = Tk()
        self.root.title("Paint Window")
        self.window_title = "Paint Window"
        self.model = model
        self.model_folder = model_folder
        self.transform = transform

        self.pen_button = Button(self.root, text='pen', command=self.use_pen)
        self.eraser_button = Button(self.root, text='eraser', command=self.use_eraser)
        self.clear_button = Button(self.root, text='clear', command=self.clear_canvas)
        self.predict_button = Button(self.root, text='predict', command=self.predict)
        self.choose_size_button = Scale(self.root, from_=2, to=10, orient=HORIZONTAL)
        self.dev_button = Button(self.root, text='dev', command=self.toggle_dev)

        self.pen_button.grid(row=0, column=0, padx=5, pady=5, sticky='e')
        self.eraser_button.grid(row=0, column=1, padx=5, pady=5, sticky='w')
        self.clear_button.grid(row=0, column=2, padx=5, pady=5, sticky='w')
        self.predict_button.grid(row=0, column=3, padx=5, pady=5, sticky='w')
        self.dev_button.grid(row=0, column=4, padx=5, pady=5, sticky='e')
        self.choose_size_button.grid(row=0, column=5, padx=5, pady=5, sticky='w')
        
        # PYTORCH STUFF HERE
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = self.model.to(self.device)
        self.model.load_state_dict(torch.load(self.model_folder, map_location=self.device, weights_only=True))
        self.model.eval()
        self.softmax = nn.Softmax(dim=1)

        # Canvas and output area and label
        self.f = Frame(self.root, bg='#9CAF88')
        #self.output_frame = Canvas(self.root, bg='lightgrey')  # Placeholder for output area
        self.canvasDict = {}
        self.CC = None # current canvas
        self.labelList = []
        self.dev = True
        self.storedSymbolDict = {} 

        # Arrange canvas and output area in a 2-column layout
        self.f.grid(row=1, column=0, sticky='nsew', padx=0, pady=0)
        #self.output_frame.grid(row=1, column=1, sticky='nsew', padx=0, pady=0)
        
        # Configure grid to adjust sizes
        self.root.grid_rowconfigure(1, weight=1)
        self.root.grid_columnconfigure(0, weight=3)  
        #self.root.grid_columnconfigure(1, weight=1)

        self.root.attributes('-fullscreen', True)  # Fullscreen mode
        self.root.after(150, self.setup)
        self.root.bind("<Escape>", self.exit_fullscreen)  # Bind Escape to exit fullscreen
        self.root.bind('<Control-z>', self.undo)
        self.root.bind('<Q>', self.close_window)
        self.root.bind("<Button-1>", self.getCC) 
        self.root.after(200, self.print_window_size)
        self.root.mainloop()

    def setup(self):
        self.old_x = None
        self.old_y = None
        self.line_width = self.choose_size_button.get()
        self.color = self.DEFAULT_COLOR
        self.eraser_on = False
        self.active_button = self.pen_button
        self.objects = []
        self.screenshot_img = None
        frameW, frameH = self.f.winfo_width(), self.f.winfo_height()
        for i in range(5):
            canvas = Canvas(self.f, bg='white', width=frameW+425, height=frameH//5.2)  
            canvas.bind('<B1-Motion>', self.paint)
            canvas.bind('<ButtonRelease-1>', self.reset)
            canvas.grid(row=i, column=0, padx=5, pady=5, sticky='nsew')
            self.canvasDict[i] = canvas
        self.CC = self.canvasDict[0]
        
        print(f"canvasDict: {self.canvasDict}")     

    def toggle_dev(self):
        self.dev = not self.dev

    def getCC(self, event):
        cc = event.widget
        if isinstance(cc, Canvas):
            self.CC = cc

    def exit_fullscreen(self, event=None):
        self.root.attributes('-fullscreen', False)
        self.root.quit()

    def use_pen(self):
        self.activate_button(self.pen_button)

    def choose_color(self):
        self.eraser_on = False
        self.color = askcolor(color=self.color)[1]

    def use_eraser(self):
        self.activate_button(self.eraser_button, eraser_mode=True)

    def activate_button(self, some_button, eraser_mode=False):
        self.active_button.config(relief=RAISED)
        some_button.config(relief=SUNKEN)
        self.active_button = some_button
        self.eraser_on = eraser_mode

    def clear_canvas(self):
        self.CC.delete('all')
        self.objects.clear()
        for label in self.labelList:
            label.destroy()
        self.labelList.clear()
        canvasIdx = f"{self.CC}"[-1]
        canvasIdx = 1 if canvasIdx == 's' else int(canvasIdx)
        self.storedSymbolDict = {}
        
    def paint(self, event):
        self.line_width = self.choose_size_button.get()
        paint_color = "white" if self.eraser_on else (self.color if self.color else self.DEFAULT_COLOR)  
        if self.old_x and self.old_y:
            obj_id = self.CC.create_line(self.old_x, self.old_y, event.x, event.y,
                               width=self.line_width, fill=paint_color,
                               capstyle=ROUND, smooth=TRUE, splinesteps=36)
            self.objects.append(obj_id)  # Store the object's ID for undo
        self.old_x = event.x
        self.old_y = event.y
    
    def copy_to_clipboard(self,string):
        # Clear the clipboard
        self.root.clipboard_clear()
        # Append new text to the clipboard
        self.root.clipboard_append(string)
        # Optional: Show a message or feedback to the user
        print("Text copied to clipboard!")

    def reset(self, event):
        self.old_x, self.old_y = None, None

    def undo(self, event=None):
        if self.objects:
            # Remove the last object added to the canvas
            last_object = self.objects[max(0,len(self.objects)-50):len(self.objects)]
            for item in last_object:
                self.CC.delete(item)
            self.objects = self.objects[0: len(self.objects)-50]

    def get_window_size(self):
        width = self.root.winfo_width()
        height = self.root.winfo_height()
        return width, height

    def print_window_size(self):
        size = self.get_window_size()
        print(f"Window size: {size}")

    def close_window(self, event=None):
        self.root.destroy()

    def bounding_box(self, x, y, width, height, color):
        self.CC.create_line(x,y,x,y+height, fill="green", width=5)
        self.CC.create_line(x,y+height,x+width,y+height, fill="green", width=5)
        self.CC.create_line(x+width,y+height,x+width,y, fill="green", width=5)
        self.CC.create_line(x+width, y, x, y, fill="green", width=5)

    def predict(self):
        # Hide the button grid
        self.pen_button.grid_forget()
        self.eraser_button.grid_forget()
        self.clear_button.grid_forget()
        self.predict_button.grid_forget()
        self.choose_size_button.grid_forget()

        self.root.update()  # Force update
        self.root.update_idletasks()  # Ensure all events are processed

        with mss.mss() as sct:
            canvas_window = {
                "top": self.CC.winfo_rooty(), 
                "left": self.CC.winfo_rootx(), 
                "width": self.CC.winfo_width(), 
                "height": self.CC.winfo_height() 
            }
            entire_ss = sct.grab(canvas_window) 

        imageBGR = cv2.cvtColor(np.array(entire_ss), cv2.COLOR_BGRA2BGR) # np array and bgr for cv2
            

        # Show the button grid again
        self.pen_button.grid(row=0, column=0, padx=5, pady=5, sticky='e')
        self.eraser_button.grid(row=0, column=1, padx=5, pady=5, sticky='w')
        self.clear_button.grid(row=0, column=2, padx=5, pady=5, sticky='w')
        self.predict_button.grid(row=0, column=3, padx=5, pady=5, sticky='w')
        self.dev_button.grid(row=0, column=4, padx=5, pady=5, sticky='e')
        self.choose_size_button.grid(row=0, column=5, padx=5, pady=5, sticky='w')

        self.run_inference(symList=bb_ss_list, labelPosList=labelPosList)

    def run_inference(self, symList, labelPosList):
        sympyList = []
        ls = self.largestSquare[2]
        for idx, syms in enumerate(symList): # these are mss screenshots
            image_array = np.array(syms[0])
            if syms[-1] <= 60: # hard code this first
                thickness_factor = 3  
            elif syms[-1] <= 125:
                thickness_factor = 2
            else: 
                thickness_factor = None # bugs at 1
            thickness_factor = None # use this to test normally
            syms.pop(-1) # get rid of side
            if thickness_factor is not None:
                kernel = np.ones((thickness_factor, thickness_factor), np.uint8)
                augmented_img_array = cv2.dilate(image_array, kernel, iterations=1)
                pil_image = Image.fromarray(augmented_img_array)
                ss_img = pil_image.resize((45,45)) # NEVER INVERT THIS, trained on B-on-W data
            else:
                ss_img = Image.fromarray(np.array(syms[0])).resize((45,45))
            
            input_tensor = self.transform(ss_img)

            if isinstance(input_tensor, torch.Tensor):   
                #showIm = np.squeeze(input_tensor.numpy()) 
                #plt.clf()
                #plt.imshow(showIm)
                #plt.show() # uncomment to debug
                
                input_tensor = torch.unsqueeze(input_tensor, 0) # Add batch dim
                #torch.set_printoptions(threshold=1000, edgeitems=10)
                #print(f"Image tensor of size {input_tensor.size()}: {input_tensor}")
                input_tensor = input_tensor.to(self.device)
            
            with torch.no_grad():
                output = self.softmax(self.model(input_tensor))
            
            output = output.squeeze(0)  # Remove batch dimension
            predictions = output.cpu().detach().numpy() if torch.cuda.is_available() else output
            predictions = list(predictions) 
            
            labels_df = class_Labels_Length('data/extracted_images_new') # imported from dataloader
            classes = labels_df['Class_Names'].tolist()

            max_prob = max(predictions)
            max_index = predictions.index(max_prob)  # Get the index of the highest prediction
            max_class_name = classes[max_index]

            '''
            plt.clf()
            plt.bar(classes, predictions, color = 'skyblue')
            plt.annotate(f'Highest: {max_class_name}',
                xy=(max_index, predictions[max_index]),
                xytext=(max_index, predictions[max_index] + 0.1),
                arrowprops=dict(facecolor='red', edgecolor='red', arrowstyle='->', lw=1.5))
            plt.show() 
            '''
        

        '''
        # split this into a left and right side of equals sign list
        canvasIdx = f"{self.CC}"[-1]
        canvasIdx = 1 if canvasIdx == 's' else int(canvasIdx)
        if '=' in sympyList:
            equalsIdx = sympyList.index('=')
            if equalsIdx == len(sympyList)-1: # nothing on right, we must solve 
                latexLeft = list_to_sympy(sympyList[0:equalsIdx])
                latexRight = solver(latexLeft, self.storedSymbolDict)
                self.solvedLabel(label_text=latexRight, x=self.equalsX + self.equalsS + 20, y=self.equalsY - 70)
                latex_str = '$' + latexLeft + '=' + latexRight + '$'
                self.convert_latex(input=latex_str)
                self.copy_to_clipboard(string=latex_str)

            else: 
                if equalsIdx == 1: # left side is single variable
                    latexLeft = list_to_sympy(sympyList[0:equalsIdx])
                    latexRight = list_to_sympy(sympyList[equalsIdx+1:len(sympyList)])
                    self.storedSymbolDict[latexLeft] = latexRight
                    print(f"storedSymbolDict: {self.storedSymbolDict}")
                    latex_str = '$' + latexLeft + '=' + latexRight + '$'
                    self.convert_latex(input=latex_str)
                    self.copy_to_clipboard(string=latex_str)

                else: 
                    latexLeft = list_to_sympy(sympyList[0:equalsIdx])
                    latexRight = list_to_sympy(sympyList[equalsIdx+1:len(sympyList)])
                    latex_str = '$' + latexLeft + '=' + latexRight + '$'
                    self.convert_latex(input=latex_str)
                    self.copy_to_clipboard(string=latex_str)
        else:
            latex_str = '$' + list_to_sympy(sympyList) + '$'
            self.convert_latex(input=latex_str) # RENDERS LATEX
            self.copy_to_clipboard(string=latex_str)
        '''

    def convert_latex(self, input): # *render latex
        print(f"latex string: {input}")
        fig = Figure(figsize=(5, 5))
        ax = fig.add_subplot(111)
        ax.text(0.5, 0.5, input, fontsize=50, ha='center', va='center')
        ax.axis('off')

        # Convert the Matplotlib figure to a PIL image
        buf = io.BytesIO()
        fig.savefig(buf, format='png')
        buf.seek(0)
        image = Image.open(buf)
        self.tk_image = ImageTk.PhotoImage(image)

        # Display the image on the Tkinter canvas
        self.CC.create_image(300, 150, image=self.tk_image)
        
    def createLabel(self, label_text, x, y, size):
        label = Label(self.CC, text = label_text, font=("Courier", max(size, 100)//10))
        label.place(x=x, y=y)
        self.labelList.append(label)     
    
    def solvedLabel(self, label_text, x, y):
        label = Label(self.CC, text = label_text, bg='white', font=("Bradley Hand ITC", 70))
        label.place(x=x, y=y)
        self.labelList.append(label)

if __name__ == '__main__':
    paint_app = Paint(model=Model_1(), model_folder='runs/Model_1Experiment1Epoch1Step6000.pt', transform=transform_define)