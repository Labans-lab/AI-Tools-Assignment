AI Practical Project – Machine Learning, Deep Learning & NLP
👤 Submitted by:

[Your Full Name(s)]
Course: [e.g., AI & Data Science]
Date: [DD/MM/YYYY]

🧠 Part 1: Theoretical Questions
Q1: Differences between TensorFlow and PyTorch

Answer:

TensorFlow: Excellent for production; supports TensorFlow Lite and TensorFlow Serving.

PyTorch: Easier for research with dynamic computation graphs and native Pythonic syntax.
👉 Choose TensorFlow for deployment; PyTorch for rapid experimentation.

Q2: Use Cases for Jupyter Notebooks in AI

Interactive coding & visualization during data exploration.

Model experimentation and documentation in one place.

Q3: How spaCy Enhances NLP

Provides pretrained pipelines for tokenization, POS tagging, and NER.

Far more efficient and accurate than raw Python string functions.

Easily integrates with ML pipelines.

Comparative Analysis
Feature	Scikit-learn	TensorFlow
Focus	Classical ML (regression, classification)	Deep Learning (neural networks)
Ease for Beginners	Very easy; minimal setup	Steeper learning curve
Community Support	Large, active	Very large and industry-backed
⚙️ Part 2: Practical Implementation (50%)
🧩 Task 1 – Classical ML with Scikit-learn

Dataset: Iris Species
Goal: Predict flower species using Decision Tree Classifier.

Key Steps:

Loaded and preprocessed the Iris dataset

Split into train/test sets

Trained DecisionTreeClassifier

Evaluated using accuracy, precision, and recall

Results:

Accuracy: 96%
Precision: 95%
Recall: 94%


Sample Visualization:

<img width="1763" height="531" alt="Screenshot 2025-10-22 125038" src="https://github.com/user-attachments/assets/c2185b51-073b-441f-a7b8-48b143d2041d" />


🧠 Task 2 – Deep Learning with TensorFlow

Dataset: MNIST Handwritten Digits
Goal: Build a CNN to classify digits (0–9).

Architecture Overview:

Conv2D → MaxPooling → Flatten → Dense → Output

Activation: ReLU, Softmax

Optimizer: Adam

Results:

Test Accuracy: 98.7%


Visualization:
<img width="1259" height="636" alt="Screenshot 2025-10-22 125050" src="https://github.com/user-attachments/assets/a0e28be7-f5de-4610-a935-865d8d52c280" />


💬 Task 3 – NLP with spaCy

Dataset: Amazon Product Reviews
Goal: Extract product names (NER) and determine sentiment.

Approach:

Used spacy.load("en_core_web_sm") for NER.

Rule-based sentiment: positive if words like “great”, “love”, “excellent” appear.

Example Output:
<img width="831" height="356" alt="Screenshot 2025-10-22 125102" src="https://github.com/user-attachments/assets/3cec0fb2-78de-41c9-a435-10fa84d2e661" />


Text: "I love my new Samsung phone!"
Entities: [("Samsung", "ORG")]
Sentiment: Positive 😊

🌱 Ethical Reflection

AI models should be:

Fair: Avoid bias in datasets (e.g., balanced representation).

Transparent: Explain how predictions are made.

Sustainable: Support equitable and resource-efficient solutions.

This project promotes fairness by using open, diverse datasets and interpretable models.

🎥 Presentation Video

📺 Click here to watch the 3-minute presentation

(Replace with your YouTube/Drive/Community link once uploaded)

📦 Repository Contents
File	Description
Task1_Iris_Classification.ipynb	Classical ML using Decision Tree
Task2_MNIST_CNN_TensorFlow.ipynb	CNN Model for MNIST
Task3_NLP_spaCy.ipynb	Named Entity & Sentiment Analysis
README.md	Full report and presentation
🏁 Conclusion

Through this project, we demonstrated:

Mastery of classical and deep learning workflows

Real-world NLP application using spaCy

Awareness of ethics in AI systems

"AI is not just about intelligence — it’s about responsibility." 🤖🌿
