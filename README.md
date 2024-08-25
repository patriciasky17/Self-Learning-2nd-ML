# OCR Sirekap (Election Vote Recapitulation System)

## Description
This project aims to recognize text from scanned images of vote recapitulation used in elections. The project faces challenges such as variations in handwriting and the quality of the images.

## Process Flow
1. **Detection:**
   - YOLOv8 from Ultralytics is used to detect areas containing the vote recapitulation tables in the images.
   - The detected areas are cropped to focus on important text for recognition.

2. **Text Recognition:**
   - The TrOCR model (`trocr-base-handwritten`) is used to recognize numbers and text from the table columns.
   - For text that is difficult to recognize (e.g., handwriting), preprocessing such as image binarization is applied before recognition.

3. **Evaluation:**
   - Text recognition accuracy metrics are used to evaluate the model’s performance, and the results are compared to the available ground truth data.

## Dataset
The entire dataset used in this project can be found in [Google Drive](https://drive.google.com/drive/folders/1RoPM8iAUl8a4CSjQeA_17en0qewimM4_?usp=sharing). As for model, please use training_logs (Object Detection with YOLOv8) and trocr_finetuned (Transformer-based OCR)

### Steps
1. **Image Annotation:**
   - Images are annotated using Roboflow with three classes. The dataset can be viewed on [Roboflow](https://universe.roboflow.com/braincore-3uq0a/ocr-sirekap/dataset/8).

2. **Number Detection:**
   - The YOLOv8 model is used to detect numbers in the scanned images.

3. **Text Recognition with TrOCR:**
   - The TrOCR model (`trocr-base-handwritten`) is used to identify numbers. Preprocessing such as image binarization is applied before text recognition.
   - The model is then trained to improve its accuracy in recognizing numbers.

4. **Pipeline & Optimization:**
   - All models are combined into a single pipeline. Optimization includes rotating the image if the initial detection fails to find the desired class until detection is successful.
   -The values from the `numbers_2` class are then input into a CSV file as shown in `sample_submission.csv`.

## Testing
Each step in the pipeline was tested individually to ensure the model’s quality and performance before proceeding to the next step, as you can see for the `OCR_Sirekap.ipynb` (the file might be heavy due to testing process with 200 images)

## Evaluation
Text recognition accuracy was measured using Character Error Rate (CER), and the best result achieved was 0.06 CER or approximately 6%.

### Validation Results
Here are the validation results from the model:
- Loss: 0.0459
- Character Error Rate (CER): 0.06 (sekitar 6%)
- Runtime: 41.9 detik
- Samples per Second: 6.587
- Steps per Second: 0.835
- Epoch: 3.0

## Challenges
The main challenges in this project were fine-tuning the model and preprocessing the images to improve accuracy in recognizing numbers. This project needs more exploration to achieve better accuracy.

---

Demo Video can be accessed by clicking the button below

[![Button Click]][Link]

---

[Link]: https://drive.google.com/file/d/1IVsA9MAb0c-BslE0nZmWFZ5Dgm68QlmW/view?usp=sharing
[Button Click]: https://img.shields.io/badge/Click_Me!-37a779?style=for-the-badge

<!-- [here](https://drive.google.com/file/d/1qwR_2qiS6ft4Fryv8HJbXZXcz3wZXBRt/view?usp=sharing). -->