#!/usr/bin/env python
# coding: utf-8

# In[7]:


import cv2
import sys
import pytesseract
import numpy as np
import pandas as pd
import pyodbc
import h5py
import keras
import numpy as np
import tensorflow as tf
import matplotlib.image as mpimg
from matplotlib import pyplot as plt
import datetime
import matplotlib.pyplot as plt
from mtcnn.mtcnn import MTCNN
from keras_vggface import utils
from keras_vggface import VGGFace
from PIL import Image, ImageEnhance
from keras.engine.training import  Model
from scipy.spatial.distance import cosine
from keras.layers import Flatten, Dense, Input
from keras_vggface.vggface import VGGFace
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from keras.layers import Flatten, Dense, Input,concatenate
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout
from keras.models import Model
from keras.models import Sequential
from tensorflow.keras.applications.vgg16 import VGG16


pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

def initial():
    count_front = 0
    global count
    count = 0
    global fail_counter
    fail_counter = 0
    # data_check variable will check the data validity
    global data_check 
    data_check = {}
    # final_data variable will have the data details as per approved or not approved
    global final_data 
    final_data = {}
    global simlarity_score
    simlarity_score = {}

#------------------------- Face Similarity Starts Here -------------------------------------

def extract_face(filename, required_size=(224, 224)):
    img = plt.imread(filename)
    detector = MTCNN()
    results = detector.detect_faces(img)
    x1, y1, width, height = results[0]['box']
    x2, y2 = x1 + width, y1 + height
    face = img[y1:y2, x1:x2]
    image = Image.fromarray(face)
    image = image.resize(required_size)
    enhancer = ImageEnhance.Sharpness(image)
    image = enhancer.enhance(0.5)
    face_array = np.asarray(image)
    return face_array

def get_embeddings(filenames):
    faces = [extract_face(f) for f in filenames]
    samples = np.asarray(faces, 'float32')
    samples = utils.preprocess_input(samples, version=2)
    model = VGGFace(model='senet50', include_top=False, input_shape=(224, 224, 3), pooling='max')
    pred = model.predict(samples)
    return pred

def is_match(known_embedding, candidate_embedding, thresh=0.5):
    global count
    global simlarity_score
    score = cosine(known_embedding, candidate_embedding)
    simlarity_score["Face_Similarity_Score"] = ((1-score) * 100)

    if (1-score) > 0.5:
        notes = "Face Matched"
        data_check["Face_Validation"] = "Validated"
        final_data["Face_Validation_status"] = "OK"
        count = count + 1
    else:
        notes = "Face Not Matched"
        data_check["Face_Validation"] = "Not Validated"
        final_data["Face_Validation_status"] = "Not OK"
    return  notes

def face_similarity_executor(adhaar_front_path, user_image_path):

    embeddings = get_embeddings([adhaar_front_path, user_image_path])
    return print(is_match(embeddings[0], embeddings[1]))

#------------------------- Face Similarity Ends Here ----------------------------------------


#------------------------- Emblem Detection Starts Here -------------------------------------

def Emblem_Extractor(img, template):
    height, width = img.shape[:2]
    image_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    w, h = template.shape[::-1]
    result = cv2.matchTemplate(image_gray, template, cv2.TM_CCOEFF_NORMED)
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)
    top_left = max_loc
    bottom_right = (top_left[0] + w, top_left[1] + h)
    cv2.rectangle(img, top_left, bottom_right, (0,0,255), 1)
    cv2.imwrite(r'adhaar_emblem.jpg', img)

    x = top_left[0]
    y = top_left[1]
    w = bottom_right[0] - top_left[0]
    h = bottom_right[1] - top_left[1]

    crop_img = img[y:y+h, x:x+w]
    return crop_img 

def Emblem_Validator(img, template):

    global count
    global simlarity_score

    vgg16 = VGG16(weights='imagenet', include_top=True, pooling='max', input_shape=(224, 224, 3))
    basemodel = Model(inputs=vgg16.input, outputs=vgg16.get_layer('fc2').output)

    def get_feature_vector(img):
        img = cv2.resize(img, (224, 224))
        feature_vector = basemodel.predict(img.reshape(1, 224, 224, 3))
        return feature_vector

    def calculate_similarity(vector1, vector2):
        return (1 - cosine(vector1, vector2))

    f1 = get_feature_vector(img)
    f2 = get_feature_vector(template)
    calculate_similarity(f1, f2)

    simlarity_score["Emblem_Validation_Score"] = (calculate_similarity(f1, f2) * 100)

    if calculate_similarity(f1, f2) >= 0.5:
        count = count + 1
        data_check["Emblem_Validation"] = "Validated"
        final_data["Emblem_Validation"] = "OK"
        return print("Emblem Matched")
    else:
        data_check["Emblem_Validation"] = "Not Validated"
        final_data["Emblem_Validation"] = "Not OK"
        return print("Emblem Failed to Match")

def emblem_validation_executor(path_to_image, path_to_template):

    template_1 = cv2.imread(path_to_template, 0)
    input_image = cv2.imread(path_to_image)

    input_image = Emblem_Extractor(input_image, template_1)
    template_2 = cv2.imread(path_to_template)
    return Emblem_Validator(input_image, template_2)

#------------------------- Emblem Detection Ends Here -----------------------------------

#------------------------- GOI Detection Start Here -------------------------------------

def goi_detection(front_image_path, path_to_emblem_template, path_to_goi_template):

    pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

    image_color = cv2.imread(front_image_path)
    height, width = image_color.shape[:2]
    image_gray = cv2.cvtColor(image_color, cv2.COLOR_BGR2GRAY)
    template = cv2.imread(path_to_emblem_template, 0)  # ------------------- Template Need to be loaded
    w_1, h_1 = template.shape[::-1]

    # Load template
    template_1 =  cv2.imread(path_to_goi_template)  # --------------------- Template Need to be loaded
    w_2, h_2 = template_1.shape[1],template_1.shape[0]

    # Perform template matching!
    result = cv2.matchTemplate(image_gray, template, cv2.TM_CCOEFF_NORMED)

    # Find the indices of the object to be find 
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)

    # Draw a box around the object (if found)
    top_left = (max_loc[0]+w_1,max_loc[1])
    bottom_right = (top_left[0] + w_2+100+50+10+10, top_left[1] + h_2+20)
    cv2.rectangle(image_color, top_left, bottom_right, (0,0,255), 1)

    cv2.imwrite('mine_goi.jpg', image_color)  # ----------------------- Define Working Directory to store

    # Cropping and saving
    x = top_left[0]
    y = top_left[1]
    w = bottom_right[0] - top_left[0]
    h = bottom_right[1] - top_left[1]

    crop_img = image_color[y:y+h, x:x+w]
    cv2.imwrite('crop_goi_adhar.jpg', crop_img) # ----------------------- Define Working Directory to store

    # Extraction of text GOI(if found)

    # Read image from which text needs to be extracted
    img = cv2.imread("crop_goi_adhar.jpg") # ---------------------------- Load the previously saved Image 

    # Preprocessing the image starts

    # Convert the image to gray scale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Performing OTSU threshold
    ret, thresh1 = cv2.threshold(gray, 0, 255, cv2.THRESH_OTSU | cv2.THRESH_BINARY_INV)

    # Specify structure shape and kernel size. 

    # Empty ocr_text
    ocr_text_front = ' ' 
    text_1 = ' '
    text_2 = ' '
    text_3 = ' '
    for i in range(1,10):

        rect_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (i,i))
        # dilation = rect_kernel  
        # Appplying dilation on the threshold image
        dilation = cv2.dilate(thresh1, rect_kernel, iterations = 1)

        # Finding contours
        contours, hierarchy = cv2.findContours(dilation, cv2.RETR_EXTERNAL, 
                                                         cv2.CHAIN_APPROX_NONE)

        # Creating a copy of image
        im2 = img.copy()


        for cnt in contours:
            x, y, w, h = cv2.boundingRect(cnt)

            # Drawing a rectangle on copied image
            rect = cv2.rectangle(im2, (x, y), (x + w, y + h), (0, 255, 0), 2)

            # Cropping the text block for giving input to OCR
            cropped = im2[y:y + h, x:x + w]
        #     cv2.imshow('Cropped',cropped)

            # Open the file in append mode
            file = open("recognized.txt", "a")

            # Apply OCR on the cropped image
            text_1 = pytesseract.image_to_string(cropped,lang='eng',  config='--psm 10 --oem 3 -c tessedit_char_whitelist=0123456789')
            text_2 = pytesseract.image_to_string(cropped)
            text_3 = pytesseract.image_to_string(cropped,lang='eng',config=f'-l eng --psm 6 --oem 3 ')

            # Text generation  
            ocr_text_front = text_1 + text_2 + text_3+ ocr_text_front


        if "GOVERNMENT" in ocr_text_front.upper() and "INDIA" in ocr_text_front.upper():
            break


            #Address Check
    goi_list = ["GOVERNMENT","INDIA"]
    count_goi = 0
    for item in goi_list:
        if item.upper() in ocr_text_front.upper():
            count_goi = count_goi +1


    percent_correct_goi = count_goi/len(goi_list)
    simlarity_score["GOI"] = percent_correct_goi

    global count

    if percent_correct_goi == 1:
        count = count + 1
        data_check["GOI_Validation"] = "Validated"
        final_data["GOI_Validation_status"] = "OK"
        return print("GOI Detected and validated.....")
    else:
        data_check["GOI_Validation"] = "Not Validated"
        final_data["GOI_Validation_status"] = "Not OK"
        return print("GOI NOT Detected and validated.....")

#------------------------- GOI Detection Ends Here ----------------------------------

#-------------------------Verhoff Algorithm Starts Here -----------------------------

def verhoff_algo(UID):
    verhoeff_table_d = (
        (0,1,2,3,4,5,6,7,8,9),
        (1,2,3,4,0,6,7,8,9,5),
        (2,3,4,0,1,7,8,9,5,6),
        (3,4,0,1,2,8,9,5,6,7),
        (4,0,1,2,3,9,5,6,7,8),
        (5,9,8,7,6,0,4,3,2,1),
        (6,5,9,8,7,1,0,4,3,2),
        (7,6,5,9,8,2,1,0,4,3),
        (8,7,6,5,9,3,2,1,0,4),
        (9,8,7,6,5,4,3,2,1,0))
    verhoeff_table_p = (
        (0,1,2,3,4,5,6,7,8,9),
        (1,5,7,6,2,8,3,0,9,4),
        (5,8,0,3,7,9,6,1,4,2),
        (8,9,1,6,0,4,3,5,2,7),
        (9,4,5,3,1,2,6,8,7,0),
        (4,2,8,6,5,7,3,9,0,1),
        (2,7,9,3,8,0,6,4,1,5),
        (7,0,4,6,9,1,3,2,5,8))

    verhoeff_table_inv = (0,4,3,2,1,5,6,7,8,9)

    def calcsum(number):
        """For a given number returns a Verhoeff checksum digit"""
        c = 0
        for i, item in enumerate(reversed(str(number))):
            c = verhoeff_table_d[c][verhoeff_table_p[(i+1)%8][int(item)]]
        return verhoeff_table_inv[c]

    def checksum(number):
        """For a given number generates a Verhoeff digit and
        returns number + digit"""
        c = 0
        for i, item in enumerate(reversed(str(number))):
            c = verhoeff_table_d[c][verhoeff_table_p[i % 8][int(item)]]
        return c

    def generateVerhoeff(number):
        """For a given number returns number + Verhoeff checksum digit"""
        return "%s%s" % (number, calcsum(number))

    def validateVerhoeff(number):
        """Validate Verhoeff checksummed number (checksum is last digit)"""
        return checksum(number) == 0
    global count
    first,second,third =  map(int, UID.split(' '))
    adhar_number = str(first)+str(second)+str(third)
    if validateVerhoeff(adhar_number):
        count = count+1
        return print("Adhar number is correct")

    else:
        return print("Adhar number is incorrect")

#-------------------------Verhoff Algorithm Ends Here -----------------------------

#-------------------------OCR Front & Back Text Extraction Starts Here -----------------------------


def adhar_front_text_extraction(front_image_path):

    pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe" # ------- Location of OCR
    img = cv2.imread(front_image_path)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    ret, thresh1 = cv2.threshold(gray, 0, 255, cv2.THRESH_OTSU | cv2.THRESH_BINARY_INV)
    rect_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (100,100))
    dilation = cv2.dilate(thresh1, rect_kernel, iterations = 1)

    contours, hierarchy = cv2.findContours(dilation, cv2.RETR_EXTERNAL, 
                                                     cv2.CHAIN_APPROX_NONE)
    im2 = img.copy()

    ocr_text_front = ' ' 
    text_1 = ' '
    text_2 = ' '
    text_3 = ' '

    for cnt in contours:

        x, y, w, h = cv2.boundingRect(cnt)
        rect = cv2.rectangle(im2, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cropped = im2[y:y + h, x:x + w]

        text_1 = pytesseract.image_to_string(cropped, config=f'-l eng --psm 6 --oem 3 ')
        text_2 = pytesseract.image_to_string(cropped)
        text_3 = pytesseract.image_to_string(cropped,lang='eng',  config='--psm 10 --oem 3 -c tessedit_char_whitelist=0123456789')

        ocr_text_front = text_1 + text_2 + text_3 + ocr_text_front
    return ocr_text_front

def adhar_back_text_extraction(back_image_path):

    img = cv2.imread(back_image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret, thresh1 = cv2.threshold(gray, 0, 255, cv2.THRESH_OTSU | cv2.THRESH_BINARY_INV)

    rect_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (100,100))
    dilation = cv2.dilate(thresh1, rect_kernel, iterations = 1)
    contours, hierarchy = cv2.findContours(dilation, cv2.RETR_EXTERNAL, 
                                                     cv2.CHAIN_APPROX_NONE)

    im2 = img.copy()

    ocr_text_back = ' ' 
    text_1 = ' '
    text_2 = ' '
    text_3 = ' '

    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)

        rect = cv2.rectangle(im2, (x, y), (x + w, y + h), (0, 255, 0), 2)

        cropped = im2[y:y + h, x:x + w]

        text_1 = pytesseract.image_to_string(cropped, config=f'-l eng --psm 6 --oem 3 ')
        text_2 = pytesseract.image_to_string(cropped)
        text_3 = pytesseract.image_to_string(cropped,lang='eng',  config='--psm 10 --oem 3 -c tessedit_char_whitelist=0123456789')

        ocr_text_back = text_1  + text_2 + text_3 + ocr_text_back
    return ocr_text_back

#------------------------- OCR Front & Back Text Extraction Ends Here -----------------------------
#------------------------- Extracted Text Validation Starts Here -----------------------------


def card_validation(ocr_text_front,ocr_text_back,name,date_of_birth,Gender,UID,father_name,pincode,address):

    #FRONT CARD VALIDATION    
    count_front = 0

    # data_check variable will check the data validity
    global data_check 
    # final_data variable will have the data details as per approved or not approved
    global final_data 
    global simlarity_score
    #Name check
    name_list = list(name.split(' '))
    count_name = 0
    for item in name_list:
        if item.upper() in ocr_text_front.upper():
            count_name = count_name +1

    percent_correct_name = count_name/len(name_list)
    simlarity_score["name"] = percent_correct_name
    if percent_correct_name ==1:
        print("Name is correct.....")
        print("----------------------------")
        count_front = count_front+1
        data_check["name"] = "Validated"
        final_data["name"] = name

    else:
        print("Name is incorrect.....")
        print("----------------------------")
        data_check["name"] = "Not Validated"
        final_data["name"] = "NA"
    # Date of birth check

    if date_of_birth in ocr_text_front:
        print("Date of birth is correct.....")
        print("----------------------------")
        count_front = count_front +1
        data_check["date_of_birth"] = "Validated"
        final_data["date_of_birth"] = date_of_birth
        simlarity_score["date_of_birth"] = 1
    else:
        print("Data of birth is incorrect.....")
        print("----------------------------")
        data_check["date_of_birth"] = "Not Validated"
        final_data["date_of_birth"] = "NA"
        simlarity_score["date_of_birth"] = 0

    # Gender check
    if Gender.upper() in ocr_text_front.upper():
        print("Gender is correct.....")
        print("----------------------------")
        count_front = count_front +1
        data_check["Gender"] = "Validated"
        final_data["Gender"] = Gender
        simlarity_score["Gender"] = 1
    else:
        print("Gender is incorrect.....")
        print("----------------------------")
        data_check["Gender"] = "Not Validated"
        final_data["Gender"] = "NA"
        simlarity_score["Gender"] = 0

    # Adhar card number check
    first,second,third =  map(int, UID.split(' '))

    uid_list = list(UID.split(' '))
    count_uid = 0
    for item in uid_list:
        if item.upper() in ocr_text_front.upper():
            count_uid = count_uid +1

    percent_correct_uid = count_uid/len(uid_list)
    simlarity_score["UID"] = percent_correct_uid
    if percent_correct_uid==1:
        print("Adhar number is correct.....")
        print("----------------------------")
        count_front = count_front +1
        data_check["UID"] = "Validated"
        final_data["UID"] = UID
    else:
        print("Adhar number is incorrect.....")
        print("----------------------------")
        data_check["UID"] = "Not Validated"
        final_data["UID"] = "NA"


    # BACK CARD VALIDATION    
    count_back =0

    # Initialising NO for all check

    # Father's name check
    father_list = list(father_name.split(' '))
    count_father = 0
    for item in father_list:
        if item.upper() in ocr_text_back.upper():
            count_father = count_father +1

    percent_correct_father = count_father/len(father_list)
    simlarity_score["Father_name"] = percent_correct_father
    if percent_correct_father==1:

        print("Father Name is correct.....")
        print("----------------------------")
        count_back = count_back+1
        data_check["Father_name"] = "Validated"
        final_data["Father_name"] = father_name

    else:
        print("Father Name is incorrect.....")
        print("----------------------------")
        data_check["Father_name"] = "Not Validated"
        final_data["Father_name"] = "NA"
    # Pin code Check
    if str(pincode) in ocr_text_back:
        print("Pincode is correct.....")
        print("----------------------------")
        count_back = count_back +1
        data_check["pincode"] = "Validated"
        final_data["pincode"] = pincode
        simlarity_score["pincode"] = 1
    else:
        print("Pincode is incorrect.....")
        print("----------------------------")
        data_check["pincode"] = "Not Validated"
        final_data["pincode"] = "NA"
        simlarity_score["pincode"] = 0

    #Address Check
    address_list = list(address.split(' '))
    count_address = 0
    for item in address_list:
        if item.upper() in ocr_text_back.upper():
            count_address = count_address +1

    percent_correct_address = count_address/len(address_list)
    simlarity_score["address"] = percent_correct_address
    if percent_correct_address >= 0.5:
        print("Address is correct.....")
        print("----------------------------")
        count_back = count_back+1
        data_check["address"] = "Validated"
        final_data["address"] = address
    else:
        print("Address is incorrect.....")
        print("----------------------------")
        data_check["address"] = "Not Validated"
        final_data["address"] = "NA"

    return count_front,count_back

#------------------------- Extracted Text Validation Ends Here -----------------------------
#------------------------- Card Information Validation Counter Starts Here -----------------


def status_of_card_validation(count_front,count_back):
    #FRONT
    if count_front==4:
        colored_text = colored(0,255, 0, 'FRONT APPROVED ')
        print(colored_text)

    else:
        colored_text = colored(255,0,0, 'FRONT NOT APPROVED')
        print(colored_text)
    print("---------------------")  
    #BACK
    if count_back==3:
        colored_text = colored(0,255, 0, 'BACK APPROVED')
        print(colored_text)
        print("---------------------")
    else:
        colored_text = colored(255,0,0, 'BACK NOT APPROVED')
        print(colored_text)
        print("---------------------")
    print("---------------------")
    #COMBINE
    global count
    count = count_front + count_back + count
    
    if count==11:
        colored_text = colored(0,255, 0, 'Provided details are matching with Adhar')
        print(colored_text)
        print(colored(0,255, 0, 'OVERALL APPROVED'))
        print("---------------------")
    else:
        colored_text = colored(255,0,0, 'Provided details are not matching with Adhar')
        print(colored_text) 
        print(colored(255,0, 0, 'OVERALL NOT APPROVED'))
        print("---------------------")
    print("---------------------")
    
def colored(r, g, b, text):
    return "\033[38;2;{};{};{}m{} \033[38;2;255;255;255m".format(r, g, b, text)

#------------------------- Card Information Validation Counter Ends Here -----------------  

#--------------------------UIDAI Check Starts Here ----------------------------------
def uidai_check(adhar_number):  

    conn = pyodbc.connect("Driver={SQL Server};"
                      "Server=LTI-MEAN-63\SQLEXPRESS;"
                      "Database=SOLVATHON;"
                      "Trusted_Connection=Yes;")

    cursor = conn.cursor() 


    cursor.execute("SELECT * FROM UIDAI_DATABASE WHERE [Adhaar_Number]='%s'" % adhar_number)
    global uidai_adhaar 
    global uidai_name 
    global uidai_gender 
    global uidai_father_name
    global uidai_contact 
    global uidai_dob 
    global uidai_address 
    global uidai_pincode 

    for row in cursor:
        uidai_adhaar = row[0]
        uidai_name = row[1]
        uidai_gender = row[2]
        uidai_father_name = row[3]
        uidai_contact = row[4]
        uidai_dob = row[5]
        uidai_address = row[6]
        uidai_pincode = row[7]


    global count
    global uidai_count
    
    uidai_count =0
    
    
    #adhar number
    if final_data['UID'][0] == uidai_adhaar:
        uidai_count =uidai_count+1
    else:
        print(print(colored(255,0, 0, "UIDAI ADdhaar NOT Matched------------------------------")))
    #Name
    if final_data['name'][0].upper() == uidai_name.upper():
        uidai_count =uidai_count+1
    else:
        print(print(colored(255,0, 0, "UIDAI Name NOT Matched------------------------------")))
    #gender
    if final_data['Gender'][0].upper() == uidai_gender.upper():
        uidai_count =uidai_count+1
    else:
        print(print(colored(255,0, 0, "UIDAI Gender NOT Matched------------------------------")))
    #Father
    if final_data['Father_name'][0].upper() == uidai_father_name.upper():
        uidai_count =uidai_count+1
    else:
        print(print(colored(255,0, 0, "UIDAI Father Name NOT Matched------------------------------")))
    #DOB
    if final_data['date_of_birth'][0] == uidai_dob:
        uidai_count =uidai_count+1
    else:
        print(print(colored(255,0, 0, "UIDAI DOB NOT Matched------------------------------")))
    #Pincode
    if str(final_data['pincode'][0]) == str(uidai_pincode):
        uidai_count =uidai_count+1
    else:
        print(print(colored(255,0, 0, "UIDAI Pincode NOT Matched------------------------------")))

    #Address Check
    address_list_1 = list(uidai_address.split(' '))
    count_address_1 =0 
    for item in address_list_1:
        if item.upper() in final_data['address'][0].upper():
            count_address_1 = count_address_1 +1

    percent_correct_address_1 = count_address_1/len(address_list_1)

    if percent_correct_address_1 >= 0.5:
        uidai_count =uidai_count+1
    else:
        print(print(colored(255,0, 0, "UIDAI Address NOT Matched------------------------------")))
    
    if uidai_count == 7:
        count = count +1
        print(print(colored(0,255, 0, "UIDAI Matched------------------------------")))
    else:
        print(print(colored(255,0, 0, "UIDAI NOT Matched------------------------------")))
        
    

#--------------------------UIDAI Check END Here ----------------------------------        
    
#------------------------- Driver Function Starts Here -----------------------------------

def Executor(name, date_of_birth, gender, uid, father_name, address, pincode, front_image, back_image, photo_graph, path_to_emblem_template, path_to_goi_template, appno):
    initial()
    global count
    global fail_counter
    
    #ADD SPACES IN UID
    first_part = uid[0:4]+' '
    second_part = uid[4:8]+' '
    third_part = uid[8:12]
    uid = first_part+second_part+third_part
    
    # Input to Face Similarity

    face_similarity_executor(front_image, photo_graph)

    print("---------------------------------------------------------------")

    emblem_validation_executor(front_image, path_to_emblem_template)

    print("---------------------------------------------------------------")

    goi_detection(front_image, path_to_emblem_template, path_to_goi_template)

    print("---------------------------------------------------------------")

    verhoff_algo(uid)

    print("---------------------------------------------------------------")

    ocr_text_front=adhar_front_text_extraction(front_image)
    ocr_text_back = adhar_back_text_extraction(back_image)
    validated=card_validation(ocr_text_front,ocr_text_back,name,date_of_birth,gender,uid,father_name,pincode,address)

    print("---------------------------------------------------------------")

    count_front,count_back = validated
    status_of_card_validation(count_front,count_back)

    global simlarity_score
    global final_data

    final_data = pd.DataFrame.from_dict(final_data , orient='index').T.astype("string")
    simlarity_score = pd.DataFrame.from_dict(simlarity_score , orient='index').T.astype('string')
    final_data["Face_Similarity_Score"] = simlarity_score["Face_Similarity_Score"]
    final_data["Emblem_Validation_Score"] = simlarity_score["Emblem_Validation_Score"]

    uidai_check(uid)

    if count==12:
        final_data["Authentication_Status"] = "Passed"
        final_data["Fail_Counter"] = fail_counter
        update_db(appno)
        return "KYC Successful"
    else:
        final_data["Authentication_Status"] = "Failed"
        fail_counter = fail_counter + 1
        final_data["Fail_Counter"] = fail_counter
        update_db(appno)
        return "KYC Unsuccessful"


def update_db(appno):
    
    # global simlarity_score
    # global final_data
    
    # final_data = pd.DataFrame.from_dict(final_data , orient='index').T.astype("string")
    # simlarity_score = pd.DataFrame.from_dict(simlarity_score , orient='index').T.astype('string')
    # final_data["Face_Similarity_Score"] = simlarity_score["Face_Similarity_Score"]
    # final_data["Emblem_Validation_Score"] = simlarity_score["Emblem_Validation_Score"]

    conn = pyodbc.connect("Driver={SQL Server};"
                      "Server=LTI-MEAN-63\SQLEXPRESS;"
                      "Database=SOLVATHON;"
                      "Trusted_Connection=Yes;")

    cursor = conn.cursor()
    
    for row in final_data.itertuples():
        cursor.execute('''
                UPDATE CustomerDetailsSolvathon
                SET Authentication_Status = ?,Failure_Count = ?,Face_Detection_Accuracy = ?,Emblem_Detection_Accuracy = ?
                WHERE Application_Number = ?
                ''',
                row.Authentication_Status,
                row.Fail_Counter,
                row.Face_Similarity_Score,
                row.Emblem_Validation_Score,
                appno
                )
        conn.commit()

        return "DB Updated"

#Executor('Shishir Gupta', '01/02/1999', 'MALE', '261363713327', 'Sanjay Gupta', 'BMMIG - 29 INDRAPURAM AGRA', int(282001), 'front.jpg', 'back.jpg', 'pic.jpg', 'crop_emblem_2.jpg', 'crop_goi.jpg', 'SHIS3327')

