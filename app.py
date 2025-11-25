import streamlit as st
import tensorflow as tf
import numpy as np
import cv2
from PIL import Image

# --- 1. SETUP PAGE CONFIGURATION ---
st.set_page_config(
    page_title="Wafer Defect Detective",
    page_icon="🔍",
    layout="centered"
)

# --- 2. LOAD MODELS (Cached so they don't reload every time) ---
@st.cache_resource
def load_models():
    # Load the Gatekeeper (Good vs Bad)
    gatekeeper = tf.keras.models.load_model('gatekeeper_model.h5')
    # Load the Specialist (Which Defect?)
    specialist = tf.keras.models.load_model('my_wafer_model.h5')
    return gatekeeper, specialist

# Load them now
try:
    gatekeeper_model, specialist_model = load_models()
    st.success(" AI Models Loaded Successfully!")
except Exception as e:
    st.error(f"Error loading models: {e}")
    st.stop()

# --- 3. DEFINE LABELS ---
defect_labels = [
    'Center', 'Donut', 'Edge-Loc', 'Edge-Ring', 
    'Loc', 'Random', 'Scratch', 'Near-full', 'none'
]

# --- 4. PREPROCESSING FUNCTION ---
def preprocess_image(image_file):
    # Convert uploaded file to an OpenCV Image
    file_bytes = np.asarray(bytearray(image_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, 1) # 1 = Load as Color (though our data is 2D-ish)
    
    # 1. Resize to 64x64 (Same as training)
    img_resized = cv2.resize(img, (64, 64), interpolation=cv2.INTER_CUBIC)
    
    # 2. Convert to Grayscale (if it's not already)
    # Our model expects (64, 64, 1)
    if len(img_resized.shape) == 3:
        img_gray = cv2.cvtColor(img_resized, cv2.COLOR_BGR2GRAY)
    else:
        img_gray = img_resized

    # 3. Normalize (Divide by 2.0, NOT 255)
    img_norm = img_gray.astype('float32') / 2.0
    
    # 4. Reshape for Model (1, 64, 64, 1)
    img_final = img_norm.reshape(1, 64, 64, 1)
    
    return img_final, img_resized # Return processed & original for display

# --- 5. THE WEBSITE UI ---
st.title("🔍 Wafer Fault Detection System")
st.write("Upload a wafer map image to check for defects.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    # Process the image
    processed_img, original_img = preprocess_image(uploaded_file)
    
    # Display the User's Image
    st.image(original_img, caption="Uploaded Wafer Map", width=300)
    
    st.write("---")
    st.write("Analysis Result:")
    
    # --- STAGE 1: GATEKEEPER ---
    with st.spinner('Checking wafer quality...'):
        is_bad_prob = gatekeeper_model.predict(processed_img, verbose=0)[0][0]

    # Logic: If probability > 0.5, it is BAD.
    if is_bad_prob < 0.5:
        # WAFER IS GOOD
        confidence = (1 - is_bad_prob) * 100
        st.success(f"✅ **PASSED:** This wafer is **GOOD**.")
        st.metric(label="Confidence", value=f"{confidence:.2f}%")
        
    else:
        # WAFER IS BAD -> CALL SPECIALIST
        st.warning(f"⚠️ **FAILED:** Defect Detected! Analyzing pattern...")
        
        # --- STAGE 2: SPECIALIST ---
        defect_probs = specialist_model.predict(processed_img, verbose=0)
        defect_index = np.argmax(defect_probs)
        defect_name = defect_labels[defect_index]
        confidence = np.max(defect_probs) * 100
        
        st.error(f"❌ **Defect Type:** {defect_name}")
        st.metric(label="Confidence", value=f"{confidence:.2f}%")
        
        # Optional: Show the probability bar chart
        st.bar_chart(defect_probs.T)