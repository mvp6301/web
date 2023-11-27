# -*- coding: utf-8 -*-
"""
Created on Wed Nov 22 17:28:07 2023

@author: chaitanya
"""
from flask import Flask, render_template, request, redirect, url_for
from googletrans import Translator
from gtts import gTTS
import os
import cv2
import pytesseract
from PyPDF2 import PdfReader
from docx import Document
from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.lsa import LsaSummarizer
import io
import base64
import nltk
nltk.download('punkt')

app = Flask(__name__)

# Set the path to the Tesseract executable (update this path based on your installation)
pytesseract.pytesseract.tesseract_cmd = r"F:\New folder\tesseract.exe"

def extract_text_from_image(scanned_image):
    gray_image = cv2.cvtColor(scanned_image, cv2.COLOR_RGB2GRAY)
    text = pytesseract.image_to_string(gray_image)
    print("Extracted Text from Image:")
    print(text)
    return text

def extract_text_from_pdf(pdf_path):
    with open(pdf_path, 'rb') as file:
        pdf_reader = PdfReader(file)
        text = ""
        for page_num in range(len(pdf_reader.pages)):
            page = pdf_reader.pages[page_num]
            text += page.extract_text()
        print("Extracted Text from PDF:")
        print(text)
        return text

def extract_text_from_text_file(text_file_path):
    with open(text_file_path, 'r') as file:
        text = file.read()
        print("Text from Text File:")
        print(text)
        return text

def extract_text_from_word_document(docx_path):
    doc = Document(docx_path)
    text = ""
    for paragraph in doc.paragraphs:
        text += paragraph.text + '\n'
    print("Text from Word Document:")
    print(text)
    return text

def summarize_text_sumy(text, sentences_count):
    parser = PlaintextParser.from_string(text, Tokenizer("english"))
    summarizer = LsaSummarizer()
    summary = summarizer(parser.document, sentences_count=sentences_count)
    summary_text = " ".join(str(sentence) for sentence in summary)
    print("Summary:")
    print(summary_text)
    return summary_text

def translate_text(text, target_language='en'):
    translator = Translator()
    translation = translator.translate(text, dest=target_language)
    print(translation.text)
    return translation.text

def text_to_speech(text, language='en'):
    tts = gTTS(text=text, lang=language, slow=False)
    audio_data = io.BytesIO()
    tts.write_to_fp(audio_data)
    audio_data.seek(0)
    encoded_audio = base64.b64encode(audio_data.read()).decode('utf-8')
    return encoded_audio

"""def process_camera_image(image, sentences_count, target_language, translation_choice):
    # Add logic to process camera image, extract text, and generate results
    # For now, using dummy values
    extracted_text = "Extracted text from image"
    summarized_text = summarize_text_sumy(extracted_text, sentences_count)
    translated_text = translate_text(summarized_text, target_language) if translation_choice == 'yes' else summarized_text

    # Save audio file
    audio_file_path = 'output.mp3'
    text_to_speech(translated_text, language=target_language, save_path=audio_file_path)

    return extracted_text, summarized_text, translated_text, audio_file_path"""

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/process', methods=['POST'])
def process():
    user_choice = request.form.get('user_choice')
    if user_choice == 'camera':
        return redirect(url_for('camera'))
    elif user_choice == 'file':
        return redirect(url_for('file'))
    else:
        return "Invalid source option. Please choose 'camera' or 'file'."

@app.route('/camera')
def camera():
    return render_template('camera.html')



@app.route('/camera_processing', methods=['GET', 'POST'])
def camera_processing():
    if request.method == 'POST':
        sentences_count = int(request.form.get('sentences_count'))
        target_language = request.form.get('target_language')
        translation_choice = request.form['translation_choice']
        cap = cv2.VideoCapture(0)

        if not cap.isOpened():
            return "ERROR: Could not open the camera"

        while True:
            ret, frame = cap.read()
            cv2.imshow("Camera", frame)

            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('c'):  # Press 'c' to capture an image
                cv2.imwrite("scanned_image.png", frame)
                print("Image captured successfully as 'scanned_image.png'.")
                scanned_image = cv2.imread("scanned_image.png")
                scanned_image = cv2.cvtColor(scanned_image, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB
                text = extract_text_from_image(scanned_image)
                
                break

        # Display the extracted text on the result page
        summarized_text = summarize_text_sumy(text, sentences_count)
        translated_text = translate_text(summarized_text, target_language) if translation_choice == 'yes' else text

        # Generate audio data
        audio_data = text_to_speech(translated_text, language=target_language)

        return render_template('result.html', extracted_text=text, summarized_text=summarized_text, translated_text=translated_text, audio_data=audio_data)
    else:
        return "Error: No file provided."


    return render_template('camera.html')

@app.route('/file')
def file():
    return render_template('file.html')

# ...

@app.route('/process_file', methods=['POST'])
def process_file():
    file = request.files['file']
    sentences_count = int(request.form['sentences_count'])
    target_language = request.form['target_language']
    translation_choice = request.form['translation_choice']

    if file:
        # Save the file temporarily
        file_path = "temp_file" + os.path.splitext(file.filename)[-1]
        file.save(file_path)

        file_extension = os.path.splitext(file_path)[-1].lower()

        if file_extension == '.pdf':
            text = extract_text_from_pdf(file_path)
        elif file_extension == '.txt':
            text = extract_text_from_text_file(file_path)
        elif file_extension == '.docx':
            text = extract_text_from_word_document(file_path)
        else:
            os.remove(file_path)  # Remove the temporary file
            return f"Unsupported file format: {file_extension}"

        os.remove(file_path)  # Remove the temporary file

        summarized_text = summarize_text_sumy(text, sentences_count)
        translated_text = translate_text(summarized_text, target_language) if translation_choice == 'yes' else text

        # Generate audio data
        audio_data = text_to_speech(translated_text, language=target_language)

        return render_template('result.html', extracted_text=text, summarized_text=summarized_text, translated_text=translated_text, audio_data=audio_data)
    else:
        return "Error: No file provided."


if __name__ == "__main__":
    app.run(host="0.0.0.0",port="8080")
    



