# README

## Overview

This repository contains scripts and instructions for converting Pascal VOC format annotations to YOLOv8 format, training object detection models, and performing inference on new images using the trained models. The dataset provided contains images and annotations for detecting persons and personal protective equipment (PPE) such as hard-hats, gloves, masks, glasses, boots, vests, PPE-suits, ear protectors, and safety harnesses.

## Dataset

The dataset is provided in a zip file named `Datasets.zip`. It contains two directories:
- `images`: Contains the images for training.
- `annotations`: Contains Pascal VOC format XML files for annotations.

Additionally, a `classes.txt` file is provided for class mapping.

## Tasks

1. Convert Pascal VOC annotations to YOLOv8 format.
2. Train a YOLOv8 model for person detection.
3. Train another YOLOv8 model for PPE detection using cropped images.
4. Perform inference using both models and save the results.
5. Report the approach, learning, and evaluation metrics.

## Important Instructions

- Train a person detection model on the whole images.
- For PPE detection, train another model on cropped images of detected persons.
- Make suitable assumptions regarding class filtering and balancing for optimized solutions.
- Ensure at least 5 classes are included in the PPE detection model, even if some classes are dropped due to inconsistent results.
- Zip all the files and submit as instructed.

### Dataset Link

[Dataset Download](https://drive.google.com/file/d/1myGjrJZSWPT6LYOshF9gfikyXaTCBUWb/view?usp=sharing)

## Scripts

### 1. Pascal VOC to YOLOv8 Conversion Script

`pascalVOC_to_yolo.py`

This script converts Pascal VOC format annotations to YOLOv8 format.

#### Usage
```bash
python pascalVOC_to_yolo.py --input_dir <input_directory_path> --output_dir <output_directory_path>
```

#### Arguments
- `input_dir`: Path to the base input directory containing Pascal VOC annotations.
- `output_dir`: Path to the output directory where YOLOv8 annotations will be saved.

### 2. Inference Script

`inference.py`

This script performs inference on images using the trained YOLOv8 models for person and PPE detection.

#### Usage
```bash
python inference.py --input_dir <input_directory_path> --output_dir <output_directory_path> --person_det_model <person_detection_model_path> --ppe_detection_model <ppe_detection_model_path>
```

#### Arguments
- `input_dir`: Directory containing input images for inference.
- `output_dir`: Directory where the inference results will be saved.
- `person_det_model`: Path to the trained person detection model.
- `ppe_detection_model`: Path to the trained PPE detection model.

## Model Training

Train the YOLOv8 object detection models as per the [Ultralytics YOLOv8 documentation](https://docs.ultralytics.com/).

### Person Detection Model

- Train a model to detect persons in the images.

### PPE Detection Model

- Train a model to detect PPE items in cropped images of detected persons.

## Drawing Predicted Bounding Boxes

Use OpenCV's `cv2.rectangle()` and `cv2.putText()` functions to draw bounding boxes and confidence scores on the images.

## Report

Include a report in PDF format that details your approaches, learning experiences, and evaluation metrics. 

## Submission

- **pascalVOC_to_yolo.py**: Script for converting annotations.
- **weights/**: Directory containing the trained model weights.
- **inference.py**: Script for performing inference.
- **report.pdf**: Report detailing approaches, learning, and evaluation metrics.
- **dataset.zip**: Sample whole image and cropped image.

Ensure that all files are zipped together and submitted as per the instructions.

## Final Notes

Follow best practices for model training and scripting to achieve optimal trade-offs between speed and accuracy.
