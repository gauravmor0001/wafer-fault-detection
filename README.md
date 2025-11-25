🏭 Automated Wafer Fault Detection System

A Deep Learning system designed to automate quality control in semiconductor manufacturing. This project identifies defective silicon wafers and classifies the specific type of defect (e.g., Scratch, Edge-Ring) using a two-stage AI pipeline.

🧠 The Big Picture

In chip manufacturing, finding defects early saves millions of dollars. Manual inspection is slow and prone to error. Our system automates this process by acting as two separate "experts":

The Gatekeeper (Stage 1): A fast filter that looks at a wafer map and decides: "Is this clean or defective?" It catches the bad wafers and lets the good ones pass instantly.

The Specialist (Stage 2): If a wafer is flagged as defective, this model steps in to diagnose the exact problem (e.g., "This is a mechanical scratch" vs. "This is an etching error").

📊 How it Works

I trained our models on the WM-811K dataset (811,457 wafer maps).

Input: A wafer map image (visual representation of chip pass/fail status).

Preprocessing: I resized all maps to 64x64 and normalize the data to ensure the AI sees clear patterns instead of random noise.

Architecture:

Gatekeeper: A lightweight Binary CNN trained on a balanced dataset (50% Good, 50% Bad).

Specialist: A deeper CNN enhanced with Squeeze-and-Excitation (SE) Attention Blocks, allowing it to focus on specific defect shapes while ignoring background noise.

🚀 Results

I achieved high reliability across both stages of the pipeline:

Model           Task              Accuracy                  Key Metric

Gatekeeper  Good vs. Bad           93.3%      Low False Negative Rate (rarely misses a bad wafer)
Specialist  8 Defect Types        ~92.0%      High precision on complex shapes like Edge-Rings

🛠️ How to Run This Project

Option 1: Try the Live Demo

https://wafer-fault-detection.streamlit.app/ , click here to open the web app in your browser.

Option 2: Run Locally

If you want to run this on your own machine:

Clone the repository

git clone [https://github.com/YOUR_USERNAME/YOUR_REPO_NAME.git](https://github.com/YOUR_USERNAME/YOUR_REPO_NAME.git)
cd YOUR_REPO_NAME


Install the dependencies

pip install -r requirements.txt


Run the App

streamlit run app.py


📂 Project Structure

app.py - The main Streamlit application code.

gatekeeper_model.h5 - The trained binary classifier (Good vs. Bad).

my_wafer_model.h5 - The trained multi-class specialist (Defect Classifier).

requirements.txt - List of libraries needed to run the app.


🔮 Future Scope

I plan to improve this system by:

Adding Spatial Attention (CBAM) to better localize tiny defects like micro-scratches.

Moving from simple classification to Segmentation (U-Net), allowing the AI to draw a bounding box around the exact defect location.
