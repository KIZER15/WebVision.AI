from flask import Flask, request, jsonify
import os
import google.generativeai as genai
from PIL import Image
import io
import base64
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager
import imagehash
import shutil
from flask_cors import CORS
import logging

# Set up logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

# Initialize Flask app
app = Flask(__name__)
CORS(app)

# Configure Gemini API
API_KEY = "AIzaSyDD8QW1BggDVVMLteDygHCHrD6Ff9Dy0e8"
genai.configure(api_key=API_KEY)
#model = genai.GenerativeModel("gemini-1.5-pro")
model = genai.GenerativeModel("gemini-2.5-flash-preview-04-17")

# Configuration
SECTIONS = ['header', 'body', 'footer']
BASE_PATH = r'./'
INPUT_PATH = os.path.join(BASE_PATH, 'input')
DATASET_PATH = BASE_PATH
BEST_IMAGES_PATH = os.path.join(BASE_PATH, 'best_images')
TEMP_IMAGE_PATH = os.path.join(BASE_PATH, 'temp.png')

# Ensure directories exist
for section in SECTIONS:
    os.makedirs(os.path.join(INPUT_PATH, section), exist_ok=True)
    os.makedirs(os.path.join(DATASET_PATH, section), exist_ok=True)
os.makedirs(BEST_IMAGES_PATH, exist_ok=True)

# Compare function using perceptual hash
def get_similarity(img1_path, img2_path):
    try:
        with Image.open(img1_path) as img1, Image.open(img2_path) as img2:
            hash1 = imagehash.phash(img1)
            hash2 = imagehash.phash(img2)
        return 1 - (hash1 - hash2) / len(hash1.hash) ** 2
    except Exception as e:
        logging.error(f"Error comparing {img1_path} and {img2_path}: {e}")
        return 0

# Segment webpage function
def segment_webpage(url, segment_type, url_index):
    driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()))
    try:
        driver.get(url)
        screenshot = driver.get_screenshot_as_png()
        with open(TEMP_IMAGE_PATH, "wb") as f:
            f.write(screenshot)
        
        img = Image.open(TEMP_IMAGE_PATH)
        if segment_type == 'header':
            cropped = img.crop((0, 0, img.width, 100))
        elif segment_type == 'body':
            cropped = img.crop((0, 100, img.width, img.height - 100))
        else:
            cropped = img.crop((0, img.height - 100, img.width, img.height))
        
        output_path = os.path.join(INPUT_PATH, segment_type, f"url{url_index + 1}_{segment_type}.jpg")
        cropped.save(output_path, "JPEG")
        return output_path
    finally:
        driver.quit()

def process_urls(url_list):
    for i, url in enumerate(url_list):
        for section in SECTIONS:
            segment_webpage(url, section, i)

def find_best_matches_per_section():
    final_results = {}
    best_images = {}

    for section in SECTIONS:
        similarities_per_url = {f"url{i+1}": {'best_match': None, 'best_score': 0, 'image_path': None} for i in range(3)}

        dataset_dir = os.path.join(DATASET_PATH, section)
        if not os.path.exists(dataset_dir) or not os.listdir(dataset_dir):
            logging.warning(f"No dataset images found in {dataset_dir}")
            for i in range(3):
                similarities_per_url[f"url{i+1}"] = {'best_match': None, 'best_score': 0, 'image_path': None}
        else:
            for i in range(3):
                input_img = os.path.join(INPUT_PATH, section, f"url{i+1}_{section}.jpg")
                if not os.path.exists(input_img):
                    continue
                best_score = 0
                best_match = None

                for dataset_img in os.listdir(dataset_dir):
                    dataset_img_path = os.path.join(dataset_dir, dataset_img)
                    score = get_similarity(input_img, dataset_img_path)
                    if score > best_score:
                        best_score = score
                        best_match = dataset_img

                similarities_per_url[f"url{i+1}"]['best_match'] = best_match
                similarities_per_url[f"url{i+1}"]['best_score'] = best_score
                similarities_per_url[f"url{i+1}"]['image_path'] = input_img

        sorted_urls = sorted(similarities_per_url.items(), key=lambda x: x[1]['best_score'], reverse=True)
        final_results[section] = sorted_urls

        if sorted_urls and sorted_urls[0][1]['best_score'] > 0:
            best_image_path = sorted_urls[0][1]['image_path']
            best_images[section] = best_image_path

    for section, image_path in best_images.items():
        dest_path = os.path.join(BEST_IMAGES_PATH, f"best_{section}.jpg")
        shutil.copy(image_path, dest_path)

    # Safely clean up input directories
    for section in SECTIONS:
        input_section_path = os.path.join(INPUT_PATH, section)
        try:
            if os.path.exists(input_section_path):
                shutil.rmtree(input_section_path)
                os.makedirs(input_section_path, exist_ok=True)
        except Exception as e:
            logging.error(f"Error cleaning up {input_section_path}: {e}")

    return final_results, best_images

def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        image_data = image_file.read()
    return base64.b64encode(image_data).decode("utf-8")

def generate_suggestions(image_path, section_type, website_type="e-commerce"):
    try:
        image_base64 = encode_image(image_path)
        mime_type = "image/jpeg"

        prompt = f"""
        You are an expert in web design, user experience, and accessibility, with a deep understanding of the Web Content Accessibility Guidelines (WCAG). Analyze the {section_type} image of a {website_type} website provided below. Provide 3-5 specific, actionable suggestions to improve its design, layout, text, functionality, and accessibility. Tailor the suggestions to enhance aesthetics, usability, and accessibility for a {website_type} website, ensuring compliance with WCAG principles. Consider the following WCAG principles in your recommendations:

        - **Perceivable**: Ensure content is accessible to all users, including those using assistive technologies. Provide text alternatives for non-text content (e.g., alt text for images), ensure content can be presented in different ways without losing meaning (e.g., proper semantic structure), and make it easier for users to see and hear content (e.g., sufficient color contrast, resizable text).
        - **Operable**: Make all functionality available from a keyboard, give users enough time to read and use content, avoid content that causes seizures (e.g., flashing elements), help users navigate and find content (e.g., clear focus indicators), and make it easier to use inputs other than a keyboard (e.g., touch-friendly elements).
        - **Understandable**: Make text readable and understandable (e.g., clear language, consistent terminology), ensure content appears and operates predictably (e.g., consistent navigation), and help users avoid and correct mistakes (e.g., clear error messages).
        - **Robust**: Maximize compatibility with current and future user tools, including assistive technologies (e.g., proper HTML semantics, ARIA landmarks).

        For example, suggest changes to colors, fonts, navigation, content placement, or interactive elements that align with best practices for {website_type} websites while ensuring accessibility. Format the output as a numbered list.

        Image: [Attached]
        """

        content = [
            {"text": prompt},
            {
                "inline_data": {
                    "mime_type": mime_type,
                    "data": image_base64
                }
            }
        ]

        response = model.generate_content(content)
        suggestions_text = response.text.strip()
        suggestions_list = [line.strip() for line in suggestions_text.split("\n") if line.strip().startswith(tuple("123456789"))]
        
        return {
            "website_type": website_type,
            "section_type": section_type,
            "suggestions": suggestions_list
        }
    except Exception as e:
        logging.error(f"Error generating suggestions for {section_type}: {e}")
        return {
            "website_type": website_type,
            "section_type": section_type,
            "suggestions": []
        }

@app.route("/process-urls", methods=["POST"])
def process_urls_api():
    try:
        data = request.get_json()
        if not data or "urls" not in data:
            return jsonify({"error": "No URLs provided"}), 400

        urls = data["urls"]
        if len(urls) != 3:
            return jsonify({"error": "Exactly 3 URLs are required"}), 400

        logging.info(f"Processing URLs: {urls}")
        process_urls(urls)
        best_matches, best_images = find_best_matches_per_section()

        rankings_output = "\n🎯 Final Rankings per Section (Sorted by Best Similarity Score):\n"
        for section, rankings in best_matches.items():
            rankings_output += f"\nSection: {section.upper()}\n"
            for rank, (url, info) in enumerate(rankings, 1):
                if info['best_match']:
                    rankings_output += f"{rank}. {url} (Best Match: {info['best_match']}, Similarity: {info['best_score']:.3f})\n"
                else:
                    rankings_output += f"{rank}. {url} (No matches found in dataset)\n"

        suggestions_results = []
        for section in SECTIONS:
            if section in best_images:
                image_path = os.path.join(BEST_IMAGES_PATH, f"best_{section}.jpg")
                suggestions = generate_suggestions(image_path, section)
                suggestions_results.append(suggestions)

        return jsonify({
            "rankings": rankings_output.strip(),
            "suggestions": suggestions_results
        }), 200

    except Exception as e:
        logging.error(f"Server error: {str(e)}")
        return jsonify({"error": f"Server error: {str(e)}"}), 500

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)