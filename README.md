🩺 Disease Prediction System (NLP + Machine Learning)

This repository contains a semester project focused on predicting diseases from textual symptom descriptions using NLP, classical preprocessing, and multiple embedding-based machine learning models.
The system also includes a fully functional GUI interface for real-time symptom input and disease prediction.

📘 Project Overview
🎯 Objective

Build an intelligent system that predicts the most probable disease based on user-entered symptoms, using machine learning and biomedical language models.

🧩 Approach

The project pipeline integrates:

Robust data preprocessing

Text normalization & cleaning

Feature engineering

Vectorization using:

BioBERT embeddings

Word2Vec

FastText

GloVe

TF-IDF

Multiple trained ML models including:

Neural Networks

Hybrid embeddings + dense layers

The final system is deployed with an intuitive graphical interface to demonstrate real-time disease prediction.

🧠 Models & Embeddings Implemented
BioBERT-based Embeddings

Domain-specific biomedical language model.

Highly effective for capturing medical terminology.

Used as a powerful baseline for comparison.

Word2Vec

Semantic word embedding.

Captures contextual relationships between symptoms.

FastText

Subword-aware embedding.

Performs well with rare or misspelled symptoms.

GloVe

Global co-occurrence approach.

Lightweight and effective for structured symptom text.

Neural Network Classifier

Used as a general baseline.

Fully included in the repository.

✨ Features

Predict diseases directly from raw symptom input.

Multiple vectorization strategies for experimentation.

Training metrics & model performance tracking.

A clean, functional GUI interface for real-time predictions.

Fully reproducible machine learning pipeline.

Organized repository with executable code and demo video.

📁 Repository Contents
File / Folder	Description
notebooks/*.ipynb	Data preprocessing, vectorization, training, and analysis notebooks.
models/	Trained models, encoders, vectorizers, and training history.
data/	Dataset used for training (cleaned & raw versions).
src/	Source code for preprocessing, training, and GUI prediction scripts.
demo/	Screen recordings and screenshots of the working interface.
metrics.json	Stored accuracy, loss, and performance metrics for models.
requirements.txt	Python dependencies required for the project.
🚀 Usage
1. Clone the repository
git clone https://github.com/ZohaibShafiq3742/Disease-Prediction-System.git
cd Disease-Prediction-System

2. Install dependencies
pip install -r requirements.txt

3. Run the GUI for disease prediction
python src/predict.py

4. Run training or preprocessing

Use the available Jupyter notebooks in the notebooks/ directory.

📊 Result Highlights

Validation accuracy achieved: 98% – 100% across different embedding models.

BioBERT-based approaches showed fast convergence and stable performance.

Consistent reduction of loss across training epochs.

All results and curves are stored and visualized in notebooks.

🙌 Acknowledgements

A special thank you to:

Dr. Tanzila Kehkashan — For insightful mentorship throughout the project.

Imran Ashraf (Senior) — For continuous technical guidance and support.

📄 License

This project is released under the MIT License.

📬 Contact

Zohaib Shafiq
🔗 GitHub: https://github.com/ZohaibShafiq3742

🔗 LinkedIn: https://www.linkedin.com/in/zohaib-shafiq-33547238a/
