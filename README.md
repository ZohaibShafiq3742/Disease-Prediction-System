🩺 Disease Prediction System
Predicting diseases from symptoms using Machine Learning & NLP (BioBERT + Word2Vec + FastText + GloVe)
🚀 Project Overview
This project is an end-to-end Disease–Symptom Prediction System built using advanced NLP techniques and Machine Learning models.
The system takes symptoms as input and predicts the most probable disease using trained classification models.
The pipeline includes:
Data Preprocessing
Symptom Engineering
Text Vectorization (BioBERT, Word2Vec, FastText, GloVe, TF-IDF)
Model Training (Neural Networks & Classical ML Models)
Evaluation & Visualization
Graphical User Interface (GUI) for real-time predictions
This project demonstrates practical application of ML/NLP in the healthcare domain.
📂 Repository Structure
Disease-Prediction-System/
│── data/
│   └── Disease-Symptom-Prediction.csv
│
│── models/
│   ├── fasttext.model
│   ├── label_encoder.pkl
│   ├── tfidf_vectorizer.pkl
│   ├── model.pth
│   └── training_history.pkl
│
│── notebooks/
│   └── YourTrainingNotebook.ipynb
│
│── demo/
│   ├── interface-demo.mp4
│   └── screenshots
│
│── src/
│   ├── preprocessing.py
│   ├── train_model.py
│   └── predict.py
│
│── metrics.json  
│── README.md  
│── LICENSE  
└── requirements.txt  

🛠 Technologies & Libraries Used
Python
PyTorch
TensorFlow / Keras
Transformers (BioBERT)
Gensim (Word2Vec, FastText)
Scikit-learn
Pandas, NumPy
Matplotlib, Seaborn
NLTK
Tkinter (GUI)

⚙️ Project Workflow
1️⃣ Data Preprocessing
Cleaned disease & symptom text
Removed NaN, duplicates, and extra spaces
Combined multi-symptom fields
Normalized and standardized text

2️⃣ Feature Engineering
Label encoding
Train/validation splitting
Symptom extraction

3️⃣ Text Vectorization
We experimented with multiple embedding methods:
Embedding Model	Description
BioBERT	Domain-specific biomedical transformer
Word2Vec	Semantic vector embedding
FastText	Sub-word aware embedding
GloVe	Global co-occurrence embedding
TF-IDF	Classical but effective for sparse text
🤖 Model Training

Multiple models were trained and compared:
Neural Network Classifier
BioBERT + Dense Classifier
FastText + Classifier
GloVe + Classifier
Simple Neural Network Baseline
Training curves were logged and stored for analysis:
Loss over epochs
Accuracy over epochs

📊 Results
Across all embedding types, our models showed:
Rapid convergence
High training and validation accuracy
Smooth loss decay
Validation accuracy consistently reached:
⭐ 98% – 100% Accuracy
🖥 Graphical User Interface (GUI)

A user-friendly interface allows users to:
Enter symptoms

Process and vectorize text
Run prediction in real time
Display the predicted disease

📝 Dataset

The dataset contains:

Disease name

Corresponding symptoms

Cleaned + processed version used for training

🙌 Acknowledgements

Special thanks to:

Dr. Tanzila Kehkashan — For continuous mentorship

Imran Ashraf (Senior) — For consistent guidance & support

📬 Contact

Zohaib Shafiq
🔗 GitHub: github.com/ZohaibShafiq3742
🔗 LinkedIn: https://www.linkedin.com/in/zohaib-shafiq-33547238a/

📄 License

This project is licensed under the MIT License.
