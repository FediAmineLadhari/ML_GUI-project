import pickle
import os.path

import numpy as np
import PIL  # The Pillow library contains all the basic image processing functionality
import PIL.Image, PIL.ImageDraw
import cv2 as cv

from tkinter import *
import tkinter.messagebox
from tkinter import simpledialog, filedialog

from sklearn.svm import LinearSVC
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier


class DrawingClassifier:
    def __init__(self):
        self.triangle, self.cer, self.square = None, None, None
        self.triangle_counter, self.cer_counter, self.square_counter = None, None, None  # cer is for cercle
        self.classifier = None
        self.root = None
        self.image = None
        self.proj_name = None
        self.status_label = None
        self.canvas = None  # The Canvas is a rectangular area intended for drawing pictures or other complex layouts. You can place graphics, text, widgets or frames on a Canvas.
        self.draw = None
        self.brush_size = 15  # default brush size

        self.classes_prompt()
        self.init_gui()

    def classes_prompt(self):
        # LOADING THE PROJECT AND MAKING THE PROMPTS
        msg = Tk()
        msg.withdraw()
        self.proj_name = simpledialog.askstring("Project Name", "Enter your project name", parent=msg)

        if os.path.exists(self.proj_name):
            # IF IT EXISTS WE ONLY NEED TO LOAD THE DATA
            with open(f"{self.proj_name}/{self.proj_name}_data.pickle", "rb") as f:
                data = pickle.load(f)
            self.triangle = data["t"]
            self.cer = data["c"]
            self.square = data["s"]
            self.triangle_counter = data["tc"]
            self.cer_counter = data["cc"]
            self.square_counter = data["sc"]
            self.classifier = data["clf"]
            self.proj_name = data["pn"]
        else:
            # ELSE WE INITIALISE THE COUNTERS AND MAKE THE DIRECTORIES

            self.triangle = "triangle"
            self.square = "square"
            self.cer = "circle"

            self.triangle_counter = 1
            self.cer_counter = 1
            self.square_counter = 1

            self.classifier = LinearSVC()  # default classifier

            os.mkdir(self.proj_name)  # make directory
            os.chdir(self.proj_name)  # change directory to work on it
            os.mkdir(self.triangle)  # make a directory for each class inside of proj_name
            os.mkdir(self.cer)
            os.mkdir(self.square)
            os.chdir("..")  # get back

    def init_gui(self):
        # MAKE A ROOT AND CONNECT TO IT OUR GRAPHIC USER INTERFACE
        width, height = 500, 500  # initializing the width and height for our interface

        self.root = Tk()
        self.root.title(f"My Drawing Classifier - {self.proj_name}")

        self.canvas = Canvas(self.root, width=width - 10, height=height - 10)  # we decreased 10 to the size so the canvas won't take the whole size of the interface
        self.canvas.pack(expand=YES, fill=BOTH)  # packs our canvas widget to the gui window
        # expand − When set to true, widget expands to fill any space not otherwise used in widget's parent.
        # fill − Determines whether widget fills any extra space allocated to it by the packer, or keeps its own minimal dimensions: NONE (default), X (fill only horizontally), Y (fill only vertically), or BOTH (fill both horizontally and vertically).
        self.canvas.bind("<B1-Motion>",
                         self.paint)  # each time we click on the left button the paint fuction gets triggered

        self.image = PIL.Image.new("RGB", (width, height),
                                   (255, 255, 255))  # identifing the image class as pillow image
        self.draw = PIL.ImageDraw.Draw(self.image)

        btn_frame = tkinter.Frame(self.root)  # making a frame for the buttons
        btn_frame.pack(fill=X, side=BOTTOM)

        btn_frame.columnconfigure(0, weight=1)  # making rows for the buttons with size 1
        btn_frame.columnconfigure(1, weight=1)
        btn_frame.columnconfigure(2, weight=1)

        triangle_btn = Button(btn_frame, text=self.triangle, command=lambda: self.save(1))
        triangle_btn.grid(row=0, column=0, sticky=W + E)

        cer_btn = Button(btn_frame, text=self.cer, command=lambda: self.save(2))
        cer_btn.grid(row=0, column=1, sticky=W + E)

        square_btn = Button(btn_frame, text=self.square, command=lambda: self.save(3))
        square_btn.grid(row=0, column=2, sticky=W + E)

        bp_btn = Button(btn_frame, text="brush+", command=self.brush_plus)
        bp_btn.grid(row=1, column=0, sticky=W + E)

        clear_btn = Button(btn_frame, text="clear", command=self.clear)
        clear_btn.grid(row=1, column=1, sticky=W + E)

        bm_btn = Button(btn_frame, text="brush-", command=self.brush_minus)
        bm_btn.grid(row=1, column=2, sticky=W + E)

        train_btn = Button(btn_frame, text="train model", command=self.train_model)
        train_btn.grid(row=2, column=0, sticky=W + E)

        save_btn = Button(btn_frame, text="save model", command=self.save_model)
        save_btn.grid(row=2, column=1, sticky=W + E)

        load_btn = Button(btn_frame, text="load model", command=self.load_model)
        load_btn.grid(row=2, column=2, sticky=W + E)

        change_btn = Button(btn_frame, text="change model", command=self.change_model)
        change_btn.grid(row=3, column=0, sticky=W + E)

        predict_btn = Button(btn_frame, text="predict", command=self.predict)
        predict_btn.grid(row=3, column=1, sticky=W + E)

        save_all_btn = Button(btn_frame, text="save all", command=self.save_all)
        save_all_btn.grid(row=3, column=2, sticky=W + E)

        self.status_label = Label(btn_frame, text=f"Current model {type(self.classifier).__name__}")
        self.status_label.config(font=("Arial", 10))
        self.status_label.grid(row=4, column=1, sticky=W + E)

        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)  # Enable the close icon
        self.root.attributes("-topmost", True)  # the window will always be displayed on top of all other windows
        self.root.mainloop()  # application will continue to run until the user closes the main window or the program is terminated

    def paint(self, event):
        x1, y1 = (event.x - 1), (event.y - 1)
        x2, y2 = (event.x + 1), (event.y + 1)
        self.canvas.create_rectangle(x1, y1, x2, y2, fill='black', width=self.brush_size)
        self.draw.rectangle([x1, y1, x2 + self.brush_size, y2 + self.brush_size], fill='black', width=self.brush_size)

    def save(self, class_num):
        self.image.save("temp.png")
        img = PIL.Image.open("temp.png")
        img.thumbnail((50, 50), PIL.Image.Resampling.LANCZOS)

        if class_num == 1:
            img.save(f"{self.proj_name}/{self.triangle}/{self.triangle_counter}.png", "PNG")
            self.triangle_counter += 1
        elif class_num == 2:
            img.save(f"{self.proj_name}/{self.cer}/{self.cer_counter}.png", "PNG")
            self.cer_counter += 1
        else:
            img.save(f"{self.proj_name}/{self.square}/{self.square_counter}.png", "PNG")
            self.square_counter += 1

    def brush_plus(self):
        self.brush_size += 1

    def brush_minus(self):
        if self.brush_size > 1:
            self.brush_size -= 1

    def clear(self):
        self.canvas.delete("all")
        self.draw.rectangle([0, 0, 1000, 1000], fill='white')

    def train_model(self):
        img_list = np.array([])
        class_list = np.array([])

        for x in range(1, self.triangle_counter):
            img = cv.imread(f"{self.proj_name}/{self.triangle}/{x}.png")[:,:,0]
            img = img.reshape(2500)
            img_list = np.append(img_list, [img])
            class_list = np.append(class_list, 1)
        for x in range(1, self.cer_counter):
            img = cv.imread(f"{self.proj_name}/{self.cer}/{x}.png")[:,:,0]
            img = img.reshape(2500)
            img_list = np.append(img_list, [img])
            class_list = np.append(class_list, 2)
        for x in range(1, self.square_counter):
            img = cv.imread(f"{self.proj_name}/{self.square}/{x}.png")[:,:,0]
            img = img.reshape(2500)
            img_list = np.append(img_list, [img])
            class_list = np.append(class_list, 3)
        img_list = img_list.reshape(self.triangle_counter - 1 + self.cer_counter - 1 + self.square_counter - 1, 2500)
        self.classifier.fit(img_list, class_list)
        tkinter.messagebox.showinfo("My Drawing Classifier", "Model Trained", parent=self.root)

    def save_model(self):
        file_path = filedialog.asksaveasfilename(defaultextension="pickle")
        with open(file_path,"wb") as f:
            pickle.dump(self.classifier, f)
        tkinter.messagebox.showinfo("My Drawing Classifier", "model saved", parent=self.root)

    def load_model(self):
        file_path = filedialog.askopenfilename()
        with open(file_path, "rb") as f:
            self.classifier = pickle.load(f)
        tkinter.messagebox.showinfo("My Drawing Classifier", "model loaded", parent=self.root)

    def change_model(self):
        if type(self.classifier) == type(LinearSVC()):
            self.classifier = KNeighborsClassifier()
            print("Now using K-Nearest-Neighbors!")
        elif type(self.classifier) == type(KNeighborsClassifier()):
            self.classifier = DecisionTreeClassifier()
            print("Now using Decision Tree Classifier!")
        elif type(self.classifier) == type(DecisionTreeClassifier()):
            self.classifier = RandomForestClassifier()
            print("Now using Random Forest Classifier!")
        elif type(self.classifier) == type(RandomForestClassifier()):
            self.classifier = GaussianNB()
            print("Now using Gaussian Naive Bayes!")
        elif type(self.classifier) == type(GaussianNB()):
            self.classifier = LinearSVC()
            print("Now using Linear SVC!")

        self.status_label.config(text="Current Model: {}".format(type(self.classifier).__name__))

    def predict(self):
        self.image.save("temp.png")
        img = PIL.Image.open("temp.png")
        img.thumbnail((50,50), PIL.Image.Resampling.LANCZOS)
        img.save("predictshape.png","PNG")

        img = cv.imread("predictshape.png")[:,:,0]
        img = img.reshape(2500)
        prediction = self.classifier.predict([img])
        if prediction[0] == 1:
            tkinter.messagebox.showinfo("My Drawing Classifier", "the drawing is a triangle!")
        if prediction[0] == 2:
            tkinter.messagebox.showinfo("My Drawing Classifier", "the drawing is a circle!")
        if prediction[0] == 3:
            tkinter.messagebox.showinfo("My Drawing Classifier", "the drawing is a square!")

    def save_all(self):
        data={"t": self.triangle, "c": self.cer, "s":self.square, "tc": self.triangle_counter, "cc":self.cer_counter,
              "sc": self.square_counter, "clf": self.classifier, "pn": self.proj_name}
        with open(f"{self.proj_name}/{self.proj_name}_data.pickle", "wb") as f:
            pickle.dump(data, f)
        tkinter.messagebox.showinfo("My Drawing Classifier", "project saved", parent=self.root)

    def on_closing(self):
        answer = tkinter.messagebox.askyesnocancel("Quit", "Do you want to save your work?", parent=self.root)
        if answer is not None:
            if answer:
                self.save_all()
            self.root.destroy()
            exit()


DrawingClassifier()
