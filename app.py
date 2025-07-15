import gradio as gr
import tensorflow as tf
import numpy as np
from PIL import Image

# --- Load Model and Classes ---
def load_model_and_classes():
    """Load the pre-trained model and class names."""
    try:
        model = tf.keras.models.load_model("best_custom_cnn_model.keras", compile=False)
        class_names = ['cassava_blight', 'cassava_mosaic', 'healthy']
        return model, class_names
    except Exception as e:
        print(f"Error loading model: {e}")
        return None, None

model, class_names = load_model_and_classes()

# --- Disease Info Dictionary ---
disease_info = {
    'cassava_blight': {
        "title": "Disease Detected: Cassava Bacterial Blight (CBB)",
        "description": "Cassava Bacterial Blight is caused by the bacterium Xanthomonas axonopodis pv. manihotis and can lead to significant yield losses.",
        "symptoms": "Angular, water-soaked spots on leaves that turn brown, stem exudates, and dieback.",
        "management": "Use healthy planting materials, practice crop rotation and sanitation, prune infected parts."
    },
    'cassava_mosaic': {
        "title": "Disease Detected: Cassava Mosaic Disease (CMD)",
        "description": "Cassava Mosaic Disease is a viral disease spread by whiteflies and infected cuttings.",
        "symptoms": "Characteristic mosaic pattern on leaves with yellow-green patches and distortion.",
        "management": "Plant resistant varieties, use virus-free cuttings, rogue infected plants."
    },
    'healthy': {
        "title": "Diagnosis: Healthy",
        "description": "The leaf appears healthy and free from common diseases.",
        "symptoms": "No signs of spots, discoloration, or deformation.",
        "management": "Maintain good practices, proper nutrition, and monitor regularly."
    }
}

# --- Prediction Function ---
def predict_image(img):
    """Process image, make prediction, return formatted Markdown."""
    if img is None or model is None:
        return "<p style='color:red; font-weight:bold;'>No image provided. Please try again.</p>"

    img = img.resize((224, 224))
    img_array = np.expand_dims(np.array(img) / 255.0, axis=0)

    prediction = model.predict(img_array)
    index = np.argmax(prediction)
    confidence = prediction[0][index]
    predicted_class = class_names[index]

    info = disease_info.get(predicted_class, {
        "title": "Unknown diagnosis",
        "description": "The model could not identify the disease.",
        "symptoms": "N/A",
        "management": "N/A"
    })

    output_md = f"""
    <div style='padding: 15px; border-radius: 10px; background-color: #f9f9f9; box-shadow: 0 2px 4px rgba(0,0,0,0.1);'>
        <h3 style='color: #2d8659; font-size: 1.5em; margin-bottom: 10px;'>{info['title']}</h3>
        <p style='font-weight: bold; font-size: 1.1em;'>Confidence: {confidence:.2%}</p>
        <hr>
        <p><strong>Description:</strong> {info['description']}</p>
        <p><strong>Common Symptoms:</strong> {info['symptoms']}</p>
        <p><strong>Recommended Action:</strong> {info['management']}</p>
        <p>
            <a href='http://127.0.0.1:5000' target='_blank'
            style='display: inline-block; margin-top: 15px; padding: 10px 15px; background: #4CAF50; color: white; text-decoration: none; border-radius: 5px;'>
            Continue Diagnosis</a>
        </p>
    </div>
    """
    return output_md

# --- Custom CSS ---
css = """
body {
    background-color: #f0f8ff;
}
#main-container {
    max-width: 800px;
    margin: auto;
    padding-top: 20px;
}
#main-title {
    font-size: 2.3em;
    text-align: center;
    color: #005a34;
    font-weight: bold;
    text-shadow: 1px 1px 2px #aaa;
}
#subtitle {
    font-size: 1.2em;
    text-align: center;
    color: #555;
    margin-bottom: 25px;
}

/* Style only the Run Diagnosis button */
#run-diagnosis-btn {
    background: #4CAF50 !important;
    color: white !important;
    font-weight: bold !important;
    border-radius: 8px !important;
    transition: background-color 0.3s ease !important;
}

#run-diagnosis-btn:hover {
    background: #45a049 !important;
}

.gr-panel {
    border-radius: 15px !important;
}
"""

# --- Gradio Interface ---
with gr.Blocks(theme="gradio/soft", css=css) as demo:
    with gr.Column(elem_id="main-container"):

        gr.Markdown('<h1 id="main-title">🌿 Hybrid Decision Support System for Diagnosis of Indigenous Tuber Crops Diseases</h1>', elem_id="main-title")
        gr.Markdown('<p id="subtitle">Upload a leaf image for a preliminary diagnosis of yam and cassava diseases.</p>', elem_id="subtitle")

        with gr.Group():
            with gr.Row(equal_height=True):
                with gr.Column(scale=2):
                    image_input = gr.Image(type="pil", label="Upload Leaf Image", height=300)
                    predict_button = gr.Button("Run Diagnosis", elem_id="run-diagnosis-btn")
                with gr.Column(scale=3):
                    output_display = gr.Markdown(label="Diagnosis Results")

            gr.Examples(
                examples=[
                    "C:/Users/akinlolu/Downloads/my-disease-app/blight.jpg",
                    "C:/Users/akinlolu/Downloads/my-disease-app/mosaic.jpg",
                    "C:/Users/akinlolu/Downloads/my-disease-app/healthy.jpg"
                ],
                inputs=image_input,
                outputs=output_display,
                fn=predict_image,
                cache_examples=True
            )

    predict_button.click(fn=predict_image, inputs=image_input, outputs=output_display)

if __name__ == "__main__":
    demo.launch()
