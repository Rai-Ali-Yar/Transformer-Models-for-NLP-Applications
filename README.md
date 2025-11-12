# Transformer-Models-for-NLP-Applications
BERT-GPT2-T5-NLP-Projects

# 🧠 Transformer Models for NLP Applications

This repository contains three Natural Language Processing (NLP) tasks implemented using **state-of-the-art Transformer architectures** — BERT (Encoder-only), GPT-2 (Decoder-only), and T5 (Encoder–Decoder).  
Each task demonstrates fine-tuning, evaluation, and real-world use cases of modern NLP models using the **Hugging Face Transformers** library.

---

## 📂 Project Overview

### 🧩 Task 1: Encoder-Only (BERT) — Customer Feedback Classification
**Goal:** Classify customer feedback as **positive**, **negative**, or **neutral**.  
**Model:** BERT (Bidirectional Encoder Representations from Transformers)

- **Dataset:** [Customer Feedback Dataset (Kaggle)](https://www.kaggle.com/datasets/vishweshsalodkar/customer-feedback-dataset?select=sentiment-analysis.csv)
- **Objective:** Fine-tune BERT to understand sentiment from text.
- **Deliverables:**
  - Text preprocessing and tokenization
  - Model training and validation
  - Evaluation metrics: Accuracy, F1-Score, Confusion Matrix
  - Example predictions for unseen feedback

---

### 💻 Task 2: Decoder-Only (GPT-2) — Pseudo-code to Code Generation
**Goal:** Translate structured pseudo-code into executable Python code.  
**Model:** GPT-2 (Generative Pretrained Transformer)

- **Dataset:** [SPOC Dataset (GitHub)](https://github.com/sumith1896/spoc)
- **Research Paper:** [Sequence-to-Sequence Learning for Pseudo-Code to Code Translation (arXiv)](https://arxiv.org/pdf/1906.04908)
- **Objective:** Fine-tune GPT-2 to generate syntactically and semantically valid Python code.
- **Deliverables:**
  - Data preprocessing for pseudo-code/code pairs
  - Tokenization and formatting
  - Fine-tuning with Causal LM objective
  - Evaluation metrics: BLEU, CodeBLEU, and Human Evaluation
  - Streamlit/Gradio interface for real-time pseudo-code → code generation

---

### 📰 Task 3: Encoder–Decoder (T5/BART) — Text Summarization
**Goal:** Generate concise, meaningful summaries of news articles.  
**Model:** T5 (Text-to-Text Transfer Transformer) / BART

- **Dataset:** [CNN/DailyMail Summarization Dataset (Kaggle)](https://www.kaggle.com/datasets/gowrishankarp/newspaper-text-summarization-cnn-dailymail)
- **Objective:** Fine-tune an Encoder–Decoder model for **abstractive summarization**.
- **Deliverables:**
  - Data preprocessing (article-summary pairs)
  - Model fine-tuning using Hugging Face Transformers
  - Evaluation metrics: ROUGE-1, ROUGE-2, ROUGE-L
  - Comparison of generated summaries with ground truth

---

## ⚙️ Tech Stack

- **Language:** Python  
- **Frameworks/Libraries:**  
  - PyTorch  
  - Hugging Face Transformers  
  - Datasets, Tokenizers  
  - Scikit-learn, NLTK, NumPy, Pandas, Matplotlib  
  - Streamlit / Gradio (for interactive demo)
  
---

## 📊 Evaluation Metrics

| Task | Model | Metrics |
|------|--------|----------|
| Sentiment Classification | BERT | Accuracy, F1-score, Confusion Matrix |
| Code Generation | GPT-2 | BLEU, CodeBLEU, Human Evaluation |
| Summarization | T5/BART | ROUGE-1, ROUGE-2, ROUGE-L |

---

## 🚀 Results (Highlights)
- **BERT** achieved strong performance on customer sentiment classification.  
- **GPT-2** successfully generated syntactically correct Python code from pseudo-code.  
- **T5** produced concise summaries maintaining the semantic context of original text.

---

## 🧪 Example Outputs
**BERT Example:**  
> *Input:* "The support team was extremely helpful!"  
> *Predicted Sentiment:* Positive ✅

**GPT-2 Example:**  
> *Pseudo-code:* “for i in range 5 print i”  
> *Generated Code:*  
> ```python
> for i in range(5):
>     print(i)
> ```

**T5 Example:**  
> *Original:* “The Prime Minister announced a new economic policy...”  
> *Summary:* “PM unveils new economic strategy.”

---

## 👨‍💻 Author
**Rai Ali Yar**  
BS Computer Science — FAST University, Pakistan  
[LinkedIn](https://www.linkedin.com/in/rai-ali-yar)

---

## 🏷️ License
This project is released under the **MIT License**.
