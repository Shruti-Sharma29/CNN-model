import numpy as np
import cv2
import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
from tensorflow.keras.models import load_model

# Load trained model
model = load_model("vgg16_dog_vs_cat.h5")
classes = ["Cat", "Dog"]

def detect_objects():
    global img_label, result_label

    file_path = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg;*.png;*.jpeg")])
    if not file_path:
        return
    
    img = cv2.imread(file_path)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_resized = cv2.resize(img_rgb, (150, 150)) / 255.0
    img_array = np.expand_dims(img_resized, axis=0)
    
    prediction = model.predict(img_array)
    label = classes[int(prediction[0] > 0.5)]
    confidence = prediction[0][0] if label == "Dog" else 1 - prediction[0][0]
    
    cv2.putText(img, f"{label} ({confidence:.2f})", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_pil = Image.fromarray(img)
    img_pil = img_pil.resize((300, 300))
    tk_img = ImageTk.PhotoImage(img_pil)
    
    img_label.config(image=tk_img)
    img_label.image = tk_img
    result_label.config(text=f"Prediction: {label} ({confidence:.2f})", font=("Arial", 14, "bold"))

# GUI Setup
root = tk.Tk()
root.title("Cat vs Dog Detector")
root.geometry("400x500")

heading = tk.Label(root, text="Image Detection", font=("Arial", 16, "bold"))
heading.pack(pady=10)

btn_select = tk.Button(root, text="Choose Image", command=detect_objects, font=("Arial", 12))
btn_select.pack(pady=10)

img_label = tk.Label(root)
img_label.pack()

result_label = tk.Label(root, text="", font=("Arial", 14))
result_label.pack(pady=10)

root.mainloop()
