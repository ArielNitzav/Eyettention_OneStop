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
   ```
2. Install dependencies:
   ```sh
   pip install -r requirements.txt
   ```

## Usage
### Model Training
Execute the following command to train the model:
```sh
python main_onestop_CSD.py
```

### Inference
To apply the trained model to new data:
```sh
python main_onestop_CSD_inference.py --input_file path/to/data.csv --output_file predictions.csv
```

### Evaluation
Model performance can be assessed using standard evaluation metrics:
```sh
python evaluate.py --predictions predictions.csv --ground_truth labels.csv
```

## Dataset
The **OneStopQA dataset**, developed for eye-tracking-based readability assessment, provides fine-grained gaze-tracking annotations aligned with textual input. The dataset includes:
- Fixation duration per word
- Saccade amplitude between fixations
- Regression frequency across sentence boundaries

The dataset is preprocessed using **BERT tokenization** and **embedding extraction** to ensure compatibility with transformer-based architectures.

## Model Architecture
The model architecture is structured as follows:
- **BERT Encoder**: Generates context-sensitive word embeddings.
- **Residual Processing Layers**: Refine the encoded features for more precise prediction of gaze metrics.
- **Prediction Head**: Outputs numerical estimates for eye-tracking measures.

## Results
The trained model is evaluated based on its ability to predict eye-tracking features with minimal error. Performance is quantified using standard regression and classification metrics, depending on the specific task configuration.

## References
- [Eyettention: Leveraging Eye-tracking Data for NLP Models](https://arxiv.org/abs/2304.10784)
- [OneStopQA Dataset](https://github.com/berzak/onestop-qa)

## License
This project is released under the MIT License. See `LICENSE` for full details.
