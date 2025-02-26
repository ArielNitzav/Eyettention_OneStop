# Eyettention - OneStopQA
This repository contains an adapted implementation of the **Eyettention** model, designed to predict eye-tracking metrics from textual stimuli using the **OneStopQA dataset**. This work builds upon existing research in eye-tracking-informed NLP, specifically extending the methodologies introduced in ["Eyettention: An Attention-based Dual-Sequence Model for Predicting Human Scanpaths during Reading"](https://arxiv.org/abs/2304.10784). This adaptation generates eye reading scan paths for longer inputs (expanding from short sentences in the original paper to paragraphs) and introduces a temporal decoder for reading time durations.

For those unfamiliar with the domain, eye-tracking is the process of measuring where and for how long a person looks at different parts of a visual stimulus. In the context of **natural language processing (NLP)** and **cognitive science**, eye-tracking data provides valuable insights into human reading behavior, cognitive load, and comprehension. By modeling these patterns computationally, we can enhance applications in readability assessment, adaptive learning, and human-computer interaction.

This repository extends previous work in computational psycholinguistics by offering a reproducible and scalable framework for gaze prediction, with potential applications in readability modeling, cognitive load estimation, and adaptive learning systems.

## Overview
The model is designed to predict eye-tracking parameters—such as fixation duration, saccade length, and regression count—based on linguistic features. This approach leverages cognitive and psycholinguistic principles, integrating state-of-the-art NLP architectures. The primary contributions of this implementation include:
- **BERT-based language representation** to encode lexical and syntactic information.
- **Residual learning mechanisms** to improve predictive accuracy and mitigate vanishing gradients.
- **Fine-tuning on the OneStopQA dataset**, a corpus developed for readability assessment and gaze behavior prediction.
- **Integrated training, inference, and evaluation pipelines** to facilitate systematic experimentation.
- **Preprocessing utilities** to standardize tokenization and feature extraction.

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

## Dataset
The **OneStopQA dataset**, developed for eye-tracking-based readability assessment, provides fine-grained gaze-tracking annotations aligned with textual input. The dataset includes:
- Fixation duration per word
- Saccade amplitude between fixations
- Regression frequency across sentence boundaries

The dataset is preprocessed using **BERT tokenization** and **embedding extraction** to ensure compatibility with transformer-based architectures.

## Model Architecture
The model architecture consists of:
- **BERT Encoder**: Generates context-sensitive word embeddings.
- **Residual Processing Layers**: Refine the encoded features for more precise prediction of gaze metrics.
- **Prediction Head**: Outputs numerical estimates for eye-tracking measures.

## Results
The trained model is evaluated based on its ability to predict eye-tracking features with minimal error. Performance is quantified using standard regression and classification metrics, depending on the specific task configuration.

## References
- [Eyettention: An Attention-based Dual-Sequence Model for Predicting Human Scanpaths during Reading](https://arxiv.org/abs/2304.10784)
- [OneStopQA Dataset](https://github.com/berzak/onestop-qa)

## License
This project is released under the MIT License. See `LICENSE` for full details.

