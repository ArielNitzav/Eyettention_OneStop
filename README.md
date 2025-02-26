# Eyettention - OneStopQA

This repository presents an implementation of the **Eyettention** model, designed to predict eye-tracking metrics from textual stimuli using the **OneStopQA dataset**. This work builds upon existing research in eye-tracking-informed natural language processing (NLP), specifically extending the methodologies introduced in ["Eyettention: Leveraging Eye-tracking Data for NLP Models"](https://arxiv.org/abs/2304.10784). The adaptation of these techniques to a question-answering (QA) framework represents an effort to enhance computational models of reading behavior and cognitive load assessment.

## Overview
The model is constructed to infer eye-tracking parameters—such as fixation duration, saccade length, and regression count—based on linguistic features. This approach is informed by cognitive and psycholinguistic research, leveraging state-of-the-art NLP architectures. The primary contributions of this implementation include:
- **BERT-based language representation** to encode lexical and syntactic information.
- **Residual learning mechanisms** to improve predictive accuracy and mitigate vanishing gradients.
- **Fine-tuning on the OneStopQA dataset**, a corpus developed for readability assessment and gaze behavior prediction.
- **Integrated training, inference, and evaluation pipelines** to facilitate systematic experimentation.
- **Preprocessing utilities** to standardize tokenization and feature extraction.

This repository extends previous work in computational psycholinguistics by offering a reproducible and scalable framework for gaze prediction, with potential applications in readability modeling, cognitive load estimation, and adaptive learning systems.

## Installation
### Requirements
- Python 3.8+
- PyTorch
- Transformers (Hugging Face)
- NumPy, Pandas

### Setup
1. Clone the repository:
   ```sh
   git clone https://github.com/your-repo/Eyettention-OneStopQA.git
   cd Eyettention-OneStopQA
