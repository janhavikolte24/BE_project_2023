import streamlit as st
import torch
import detect
from PIL import Image,ImageDraw,ImageFont
import os
from datetime import datetime
import logging
import io
import sys
import contextlib
import re

logging.getLogger('tensorflow').setLevel(logging.ERROR)  # Disable TensorFlow logging

def clean_output(output):
    output = re.sub(r"(\[MODEL\].*\[\/MODEL\])", "", output)
    return output


def imageInput(device, src):
    if src == 'Upload your own data.':
        image_file = st.file_uploader("Upload An Image", type=['png', 'jpeg', 'jpg'])
        if image_file is not None:
            img = Image.open(image_file)
            st.image(img, caption='Uploaded Image', use_column_width='always')
            ts = datetime.timestamp(datetime.now())
            imgpath = os.path.join('uploads/', str(ts) + image_file.name)
            outputpath = os.path.join('outputs/', os.path.basename(imgpath))
            with open(imgpath, mode="wb") as f:
                f.write(image_file.getbuffer())
            submit = st.button("Predict!")

            # Call model prediction
            model = torch.hub.load('ultralytics/yolov5', 'custom', path='./models/best (4).pt', force_reload=True)

            model.cuda() if device == 'cuda' else model.cpu()
            model.conf = 0.15
            model.iou = 0.15
            pred = model(imgpath)
            pred = pred.xyxy[0] if len(pred) > 0 else []

            if len(pred) > 0:
                img_for_save = Image.open(imgpath)
                img_for_save.save(outputpath)
                # Display prediction
                img_ = Image.open(outputpath)
                draw = ImageDraw.Draw(img_)
                for box in pred:
                    xmin, ymin, xmax, ymax, conf, class_id = box.tolist()
                    name = model.names[int(class_id)]
                    draw.rectangle([(xmin, ymin), (xmax, ymax)], outline='yellow', width=4)
                    draw.text((xmin, ymin), name, fill='black', font=ImageFont.truetype('arial', 12))
                st.image(img_, caption='Model Prediction(s)', use_column_width='always')
                cost = calculate_cost(pred, model)
                st.write(cost)

    elif src == 'From test set.':
        # Rest of your code for selecting a test image and displaying predictions

        # Image selector slider
        test_images = os.listdir('images/')
        test_image = st.selectbox('Please select a test image:', test_images)
        submit = st.button("Predict!")
        col1, col2 = st.columns(2)

        if submit:
            with col1:
                image_file = 'images/' + test_image
                img = Image.open(image_file)
                st.image(img, caption='Selected Image', use_column_width='always')
            model = torch.hub.load('ultralytics/yolov5', 'custom', path='./models/best (4).pt', force_reload=True)
            model.cuda() if device == 'cuda' else model.cpu()
            model.conf = 0.15
            model.iou = 0.15
            pred = model(image_file)
            pred = pred.xyxy[0] if len(pred) > 0 else []

            if len(pred) > 0:
                outputpath = os.path.join('outputs/', os.path.basename(image_file))
                img_for_save = Image.open(image_file)
                img_for_save.save(outputpath)
                # Display prediction
                img_ = Image.open(outputpath)
                draw = ImageDraw.Draw(img_)
                for box in pred:
                    xmin, ymin, xmax, ymax, conf, class_id = box.tolist()
                    name = model.names[int(class_id)]
                    draw.rectangle([(xmin, ymin), (xmax, ymax)], outline='yellow', width=4)
                    draw.text((xmin, ymin), name, fill='black', font=ImageFont.truetype('arial', 18))
                with col2:
                    st.image(img_, caption='Model Prediction(s)', use_column_width='always')
                    cost = calculate_cost(pred, model)
                    st.write(cost)



def calculate_cost(pred, model):
    class_cost = {
        'Major - Front': 1000,
        'Major - Rear': 2000,
        'Major - Roof': 3000,
        'Minor - Bumper': 1000,
        'Minor - Door': 1000,
        'Minor - Front': 2000,
        'Minor - Mirror': 3000,
        'Minor - Rear': 4000,
        'Minor - Windshield': 5000,
        'Moderate - Door': 8000,
        'Moderate - Front': 10000,
        'Moderate - Rear': 11000
    }

    total_cost = 0
    severities = []

    for box in pred:
        xmin, ymin, xmax, ymax, conf, class_id = box.tolist()
        name = model.names[int(class_id)]
        if name in class_cost:
            total_cost += class_cost[name]
            severities.append(name.split(' - ')[0])

    if total_cost > 0:
        severity = max(severities, key=lambda s: ('Major' in s, 'Moderate' in s, 'Minor' in s))
        return f'Severity: {severity}, Total cost: INR {total_cost}'
    else:
        return 'No damage detected'


def main():
    # Sidebar
    st.sidebar.title('⚙️Options')
    datasrc = st.sidebar.radio("Select input source.", ['From test set.', 'Upload your own data.'])

    # if torch.cuda.is_available():
    #     deviceoption = st.sidebar.radio("Select compute Device.", ['cpu', 'cuda'], index=1)
    
    deviceoption = st.sidebar.radio("Select compute Device.", ['cpu'], index=0)
    # End of Sidebar

    st.header('Damage Detection and Cost Estimation Model')
    st.subheader('👈🏽Select the options')
    logging.getLogger('wandb').setLevel(logging.ERROR)
    st.set_option('deprecation.showPyplotGlobalUse', False)

    
    imageInput(deviceoption, datasrc)
    
    


if __name__ == '__main__':
    main()
