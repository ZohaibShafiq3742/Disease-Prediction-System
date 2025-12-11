🩺 Disease Prediction System (NLP + Machine Learning)

This repository contains a semester project focused on predicting diseases from user-provided symptom descriptions using Natural Language Processing (NLP).
Symptoms are analyzed using multiple embedding methods and machine learning models to generate accurate disease predictions.

📘 Project Overview
Objective

Predict probable diseases—based on raw textual symptom descriptions—by leveraging NLP and machine learning techniques on a medical symptom–disease dataset.

Approach

The project applies:

Classical NLP preprocessing,

Vectorization with domain-specific and general embeddings,

Neural network–based classification models
to infer disease labels from symptom text.

🧠 Models & Embeddings Implemented
BioBERT-Based Embeddings

Biomedical variant of BERT trained on PubMed.

High-quality representations for medical terminology.

Integrated into the symptom-vectorization pipeline.

Word2Vec

Learns semantic relationships between symptoms.

Provides dense numerical vectors for classification.

FastText

Handles misspellings and rare medical terms through subword modeling.

Produces robust embeddings for NLP tasks.

GloVe

Embedding approach based on global co-occurrence.

Lightweight and efficient for structured symptom data.

Neural Network Classifier

Used to compare embedding effectiveness.

Fully included in the repository.

✨ Features

Predict diseases directly from raw symptom input.

Multiple embedding methods and model architectures available.

Training and evaluation notebooks included.

Interactive GUI interface for real-time disease prediction.

Easily extendable to additional symptoms, diseases, or model variations.

📁 Repository Contents
File / Folder	Description
*.ipynb	Notebooks for preprocessing, vectorization, training, and evaluation.
models/	Trained models, encoders, vectorizers, and embedding files.
data/	Dataset used for disease–symptom prediction.
src/	Source scripts for preprocessing, training, and GUI inference.
demo/	Screen recordings and GUI demonstration files.
metrics.json	Training metrics and evaluation scores.
requirements.txt	Python dependencies required for the project.
🚀 Usage
1. Clone the repository
git clone https://github.com/ZohaibShafiq3742/Disease-Prediction-System.git
cd Disease-Prediction-System

2. Install dependencies
pip install -r requirements.txt

3. Run the GUI Application

Launch the graphical interface to input symptoms and get predictions:

python src/predict.py

4. Run notebooks

Open the .ipynb files using Jupyter Notebook or Google Colab to explore:

Preprocessing

Vectorization

Model training

Performance analysis

🩻 Inference Example

Enter symptoms such as:

fever, headache, sore throat


The system processes the text, vectorizes it, and predicts the most likely disease.

🙌 Acknowledgements

Special thanks to:

Dr. Tanzila Kehkashan — Project supervisor and mentor.

Imran Ashraf — Senior advisor for technical guidance.

📄 License

This project is released under the MIT License.

📬 Contact

Zohaib Shafiq
🔗 GitHub: https://github.com/ZohaibShafiq3742

🔗 LinkedIn: https://www.linkedin.com/in/zohaib-shafiq-33547238a/
](https://www.linkedin.com/in/zohaib-shafiq-33547238a/)
