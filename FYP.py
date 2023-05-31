import tkinter as tk
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import ImageTk, Image
from tkinter.filedialog import askopenfilename
import cv2
import os
import json
from PIL import ImageDraw


def ic_detect():
    "insert file"
    filepath = askopenfilename(
        filetypes=[  # ("image Files", "*.jpg, *.jpeg"),
            ("All Files", "*.*")]
    )
    if not filepath:
        return
    #img1 = ImageTk.PhotoImage(Image.open(filepath))
    img1 = Image.open(filepath)
    window.title(f"image1 - {filepath}")

    print(filepath)
    path_head, path_tail = os.path.split(filepath)
    file_name_only, file_type = os.path.splitext(path_tail)
    print(file_name_only)

    "Rotate image"

    MAX_FEATURES = 500
    GOOD_MATCH_PERCENT = 0.15

    def alignImages(im1, im2):

        # Convert images to grayscale
        im1Gray = cv2.cvtColor(im1, cv2.COLOR_BGR2GRAY)
        im2Gray = cv2.cvtColor(im2, cv2.COLOR_BGR2GRAY)

        # Detect ORB features and compute descriptors.
        orb = cv2.ORB_create(MAX_FEATURES)
        keypoints1, descriptors1 = orb.detectAndCompute(im1Gray, None)
        keypoints2, descriptors2 = orb.detectAndCompute(im2Gray, None)

        # Match features.
        matcher = cv2.DescriptorMatcher_create(cv2.DESCRIPTOR_MATCHER_BRUTEFORCE_HAMMING)
        matches = matcher.match(descriptors1, descriptors2, None)
        matches = list(matches)
        # Sort matches by score
        matches.sort(key=lambda x: x.distance, reverse=False)

        # Remove not so good matches
        numGoodMatches = int(len(matches) * GOOD_MATCH_PERCENT)
        matches = matches[:numGoodMatches]

        # Draw top matches
        imMatches = cv2.drawMatches(im1, keypoints1, im2, keypoints2, matches, None)
        cv2.imwrite("matches.jpg", imMatches)

        # Extract location of good matches
        points1 = np.zeros((len(matches), 2), dtype=np.float32)
        points2 = np.zeros((len(matches), 2), dtype=np.float32)

        for i, match in enumerate(matches):
            points1[i, :] = keypoints1[match.queryIdx].pt
            points2[i, :] = keypoints2[match.trainIdx].pt

        # Find homography
        h, mask = cv2.findHomography(points1, points2, cv2.RANSAC)

        # Use homography
        height, width, channels = im2.shape
        im1Reg = cv2.warpPerspective(im1, h, (width, height))

        return im1Reg, h

    if __name__ == '__main__':
        # Read reference image
        ref_file_path = 'C:/Users/60113/Desktop/FYP2/ALL_IMAGES/'
        refFilename = ref_file_path + file_name_only + '.jpg'
        print("Reading reference image : ", refFilename)
        imReference = cv2.imread(refFilename, cv2.IMREAD_COLOR)

        # Read image to be aligned
        imFilename = filepath
        print("Reading image to align : ", imFilename);
        im = cv2.imread(imFilename, cv2.IMREAD_COLOR)

        print("Aligning images ...")
        # Registered image will be resotred in imReg.
        # The estimated homography will be stored in h.
        imReg, h = alignImages(im, imReference)

        # Write aligned image to disk.
        outFilename = "aligned.jpg"
        print("Saving aligned image : ", outFilename);
        cv2.imwrite(outFilename, imReg)

        # Print estimated homography
        print("Estimated homography : \n", h)

    "load model"
    print("done")
    model = torch.hub.load('C:/Users/60113/Desktop/yolov5-master/yolov5-master', 'custom',
                           path='C:/Users/60113/Desktop/FYP/1CPP/Project1/last.pt', source='local')
    "model parameters"
    model.conf = 0.6 # confidence threshold
    model.iou = 0.3  # IoU threshold
    model.multi_label = True  # multiple labels per box
    model.classes = None  # (optional list) filter by class, i.e. = [0, 15, 16] for COCO persons, cats and dogs
    model.max_det = 10000  # maximum number of detections per image
    model.amp = False  # Automatic Mixed Precision (AMP) inference
    "ori image input"
    fileinstring = filepath  # batch of images
    fileinstring = fileinstring.replace("\\", "/")
    #toprow = tk.Frame(window)
    resize_image = img1.resize((640, 640))
    imgsized1 = ImageTk.PhotoImage(resize_image)
    ori_img = tk.Label(window, image=imgsized1)

    #toprow.pack()
    ori_img.pack(side="left")
    """process"""

    img_to_detect = fileinstring
    results = model(img_to_detect)  # custom inference size
    results.show()
    #results.save('C:/Users/60113/Desktop/FYP2/pythonProject/runs/detect/')

    # results = model(imReg)
    crops = results.crop(save=True)

    # Inference the detection result
    results.pandas().xyxy[0].to_csv('C:/Users/60113/Desktop/FYP2/pythonProject/runs/detect/' + file_name_only + '.csv')
    results.pandas().xyxy[0].to_json('C:/Users/60113/Desktop/FYP2/pythonProject/runs/detect/' + file_name_only + '.json')

    """entry pack"""
    #entry = tk.Label(background="black", foreground="white", text=results.pandas().xyxy[0])
    #entry.pack()

    """result pack"""
    ##cv2.imshow("output img", results)

    a = np.squeeze(results.render())
    # plt.imshow(a)

    # Results
    # results.show()  # results.save() or # results.show()

def coordinate_Relocation():

    "tile/header"


    "insert file"
    filepath = askopenfilename(
        filetypes=[("image Files", "*.jpg"), ("All Files", "*.*")]
    )
    if not filepath:
        return
    img1 = ImageTk.PhotoImage(Image.open(filepath))
    window.title(f"image1 - {filepath}")
    path_head, path_tail = os.path.split(filepath)
    file_name_only, file_type = os.path.splitext(path_tail)
    print(file_name_only)


    "ori image input"
    fileinstring = filepath  # batch of images
    fileinstring = fileinstring.replace("\\", "/")
    toprow = tk.Frame(window)
    ori_img = tk.Label(toprow, image=img1)
    toprow.pack()
    ori_img.pack(side="left")
    """process"""

    img_to_detect = cv2.imread(fileinstring)
    filepath_data = askopenfilename(
        filetypes=[("json file", "*.json")]
    )
    with open(filepath_data, 'r') as file:
        data = json.load(file)
    color0 = (255, 0, 0)
    color1 = (255, 255, 0)
    color2 = (255, 0, 255)
    color3 = (0, 255, 0)
    color4 = (0, 255, 255)
    color5 = (0, 0, 255)
    color6 = (20, 255, 70)
    color7 = (255, 80, 130)
    color8 = (255, 150, 30)

    color_range = [color0, color1, color2, color3, color4, color5, color6, color7, color8]

    count_array = []
    array_range = len(data['xmin'])
    for x in range(array_range):

        array_Str = ("{}".format(x))
        count_array.append(array_Str)

        xMin = data['xmin'][array_Str]
        yMin = data['ymin'][array_Str]
        xMax = data['xmax'][array_Str]
        yMax = data['ymax'][array_Str]
        class_color = data['class'][array_Str]
        start_point = (int(xMin),int(yMin))
        end_point = (int(xMax),int(yMax))

        class_name = data['name'][array_Str]
        color_range = [color0, color1, color2, color3, color4, color5, color6, color7, color8]
        thickness = 5
        img_to_detect = cv2.rectangle(img_to_detect, start_point, end_point, color_range[class_color], thickness)
        img_to_detect = cv2.putText(img_to_detect, class_name, start_point, cv2.FONT_HERSHEY_SIMPLEX, 1, color_range[class_color], thickness, cv2.LINE_AA)
        #img_to_detect_1 = ImageDraw.Draw(img_to_detect)
        #img_to_detect_1.text(start_point, class_name, fill=color_range[2])
    saving_Path = 'C:/Users/60113/Desktop/FYP2/pythonProject/runs/detect/'
    result = cv2.imwrite(saving_Path + file_name_only +"_relocated.jpg", img_to_detect)


    cv2.imshow("Coordinate Relocation Result", img_to_detect)


    if result == True:
        print("File saved successfully")
    else:
        print("Error in saving file")

    """entry pack"""
    entry = tk.Label(background="black", foreground="white")
    entry.pack()

    """result pack"""




"tile/header"
window = tk.Tk()
window.title("IC Detection")

header = tk.Label(window, text="Welcome to Jovin's FYP")
header.pack()
btn_detect = tk.Button(text="Detect", command=ic_detect)
btn_Relocate = tk.Button(text="Coordinates Relocate", command=coordinate_Relocation)
btn_detect.pack()
btn_Relocate.pack()
window.mainloop()
