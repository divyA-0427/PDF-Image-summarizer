# import os
# import re
# import cv2
# import pytesseract
# import torch
# from flask import Flask, render_template, request, jsonify
# from werkzeug.utils import secure_filename
# from pdf2image import convert_from_path
# from transformers import pipeline, logging
#
# # Suppress warnings
# logging.set_verbosity_error()
# os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
#
# # Flask app
# app = Flask(__name__)
# app.config['UPLOAD_FOLDER'] = 'uploads'
# os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
#
# # Tesseract path (adjust if needed for Linux/Mac)
# pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
#
# # Device selection
# DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
#
# # Summarizer pipeline
# SUMMARIZER_MODEL = "facebook/bart-large-cnn"
# summarizer = pipeline(
#     "summarization",
#     model=SUMMARIZER_MODEL,
#     device=0 if DEVICE == "cuda" else -1
# )
#
# # Allowed file types
# ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'bmp', 'pdf'}
#
# def allowed_file(filename):
#     return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS
#
#
# def extract_text(image_path):
#     """Extract text from an image using OCR with preprocessing + cleanup"""
#     image = cv2.imread(image_path)
#     gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#
#     # Denoise & binarize
#     gray = cv2.medianBlur(gray, 3)
#     _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
#
#     # OCR
#     text = pytesseract.image_to_string(thresh, lang="eng")
#
#     # General cleanup
#     text = re.sub(r"\n+", "\n", text)         # collapse multiple newlines
#     text = re.sub(r"\s{2,}", " ", text)       # collapse extra spaces
#     text = text.strip()
#
#     return text
#
#
# def summarize_text(text, max_length=100):
#     """Summarize text safely, remove repetition, handle long chunks"""
#     if not text.strip():
#         return "No readable text found."
#
#     # Split into chunks (to avoid token limit issues)
#     chunks = [text[i:i+1000] for i in range(0, len(text), 1000)]
#     summaries = []
#     for chunk in chunks:
#         summary = summarizer(
#             f"Summarize the following text in clear, simple English:\n\n{chunk}",
#             max_length=max_length,
#             min_length=30,
#             do_sample=False
#         )[0]['summary_text']
#         summaries.append(summary)
#
#     # Join chunks together
#     final_summary = " ".join(summaries)
#
#     # Split into sentences & deduplicate
#     sentences = re.split(r'(?<=[.!?]) +', final_summary)
#     unique = []
#     seen = set()
#     for s in sentences:
#         s_clean = s.strip()
#         if s_clean and s_clean not in seen:
#             unique.append(s_clean)
#             seen.add(s_clean)
#
#     # Rejoin cleaned summary
#     final_summary = " ".join(unique)
#
#     # Extra cleanup (remove repeated words)
#     final_summary = re.sub(r'\b(\w+)( \1\b)+', r'\1', final_summary)
#
#     return final_summary
#
#
# def process_image(image_path):
#     """Process a single image (OCR + summarization)"""
#     text = extract_text(image_path)
#     if not text:
#         return "Could not extract readable text from the image."
#     return summarize_text(text, max_length=60)
#
#
# def process_pdf(pdf_path):
#     """Convert PDF to images, OCR each page, then summarize"""
#     pages = convert_from_path(pdf_path, dpi=200)
#     full_text = ""
#
#     for i, page in enumerate(pages):
#         temp_path = f"{pdf_path}_page{i}.png"
#         page.save(temp_path, "PNG")
#         full_text += extract_text(temp_path) + " "
#         os.remove(temp_path)
#
#     if not full_text.strip():
#         return "Could not extract readable text from the PDF."
#     return summarize_text(full_text, max_length=120)
#
#
# @app.route('/')
# def index():
#     return render_template('index.html')
#
#
# @app.route('/upload', methods=['POST'])
# def upload_file():
#     if 'file' not in request.files:
#         return jsonify({'success': False, 'error': 'No file provided'})
#
#     file = request.files['file']
#     if file.filename == '':
#         return jsonify({'success': False, 'error': 'No file selected'})
#
#     if not allowed_file(file.filename):
#         return jsonify({'success': False, 'error': 'File type not allowed'})
#
#     filename = secure_filename(file.filename)
#     filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
#     file.save(filepath)
#
#     try:
#         if filename.lower().endswith('.pdf'):
#             summary = process_pdf(filepath)
#         else:
#             summary = process_image(filepath)
#
#         os.remove(filepath)
#         return jsonify({'success': True, 'summary': summary})
#
#     except Exception as e:
#         return jsonify({'success': False, 'error': str(e)})
#
#
# @app.route('/health')
# def health_check():
#     return jsonify({'status': 'healthy', 'device': DEVICE})
#
#
# if __name__ == "__main__":
#     app.run(debug=True, host="0.0.0.0", port=5000, use_reloader=False)
#



import os
import fitz  # PyMuPDF
from flask import Flask, request, jsonify, render_template
from werkzeug.utils import secure_filename
from PIL import Image
import pytesseract
from transformers import pipeline

# Initialize Flask app
app = Flask(__name__)

# Load summarization model
summarizer = pipeline("summarization", model="sshleifer/distilbart-cnn-12-6")

UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER


# ---------- Extract text ----------
def extract_text_from_pdf(pdf_path):
    text = ""
    with fitz.open(pdf_path) as doc:
        for page in doc:
            text += page.get_text("text")
    return text.strip()


def extract_text_from_image(image_path):
    img = Image.open(image_path)
    return pytesseract.image_to_string(img)


# ---------- Routes ----------
@app.route("/")
def index():
    return render_template("index.html")


@app.route("/upload", methods=["POST"])
def upload_file():
    if "file" not in request.files:
        return jsonify({"success": False, "error": "No file uploaded"})

    file = request.files["file"]
    if file.filename == "":
        return jsonify({"success": False, "error": "No file selected"})

    file_path = os.path.join(app.config["UPLOAD_FOLDER"], secure_filename(file.filename))
    file.save(file_path)

    # Extract text depending on file type
    extracted_text = ""
    if file.filename.lower().endswith(".pdf"):
        extracted_text = extract_text_from_pdf(file_path)
    elif file.filename.lower().endswith((".png", ".jpg", ".jpeg")):
        extracted_text = extract_text_from_image(file_path)
    else:
        return jsonify({"success": False, "error": "Unsupported file format"})

    if not extracted_text.strip():
        return jsonify({"success": False, "error": "No text found in file"})

    # Summarize
    try:
        summary = summarizer(extracted_text, max_length=220, min_length=100, do_sample=False)
        return jsonify({"success": True, "summary": summary[0]["summary_text"]})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)})


if __name__ == "__main__":
    app.run(debug=True)
