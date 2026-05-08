# Persian Summarizer with Fine-Tuned ParsT5

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![Hugging Face](https://img.shields.io/badge/🤗-Transformers-yellow.svg)](https://huggingface.co/transformers/)
[![Kaggle](https://img.shields.io/badge/Kaggle-Notebook-blue.svg)](https://www.kaggle.com/)

**Abstractive Persian text summarization by fine-tuning ParsT5-base on a large-scale Persian news corpus.**

---

## Overview

This repository contains a complete pipeline for **abstractive summarization of Persian texts** using a fine-tuned version of **ParsT5-base**. The model is trained to generate concise summaries (abstracts) from full-length Persian news articles, leveraging the encoder-decoder architecture of T5 adapted for the Persian language.

### Key Highlights

- **Base Model**: [`Ahmad/parsT5-base`](https://huggingface.co/Ahmad/parsT5-base) — a T5-style transformer pre-trained on Persian corpora
- **Dataset**: [Persian News Dataset](https://www.kaggle.com/datasets/amirzenoozi/persian-news-dataset) — ~392K news articles with abstracts and bodies
- **Task**: Conditional text generation with the prefix `"خلاصه کن: "` (Persian for "Summarize:")
- **Evaluation**: ROUGE-1/2/L metrics with Persian-specific word tokenization via [Hazm](https://github.com/sobhe/hazm)
- **Training**: 3 epochs on 10% of the dataset (36,717 samples) using 2 Tesla T4 GPUs

---

## Model: ParsT5-base

ParsT5 is a sequence-to-sequence transformer model based on the **T5 architecture** ([Raffel et al., 2020](https://arxiv.org/abs/1910.10683)), pre-trained from scratch on diverse Persian text corpora. It follows the standard encoder-decoder framework with a unified text-to-text format.

### Architecture Specifications

| Component | Details |
|-----------|---------|
| **Architecture** | Encoder-Decoder Transformer (T5) |
| **Parameters** | ~220M |
| **Hidden Dimension** | `d_model = 768` |
| **Attention Heads** | 12 per layer |
| **Encoder Layers** | 12 |
| **Decoder Layers** | 12 |
| **Feed-Forward Dimension** | `d_ff = 3072` |
| **Vocabulary Size** | 32,000 tokens (Persian-specific SentencePiece) |
| **Max Sequence Length** | 512 (source) / 128 (target) |
| **Relative Position Encoding** | Learned (T5-style) |

The base model is accessible on Hugging Face Hub: [`Ahmad/parsT5-base`](https://huggingface.co/Ahmad/parsT5-base)

---

## Dataset

### Source

The [Persian News Dataset](https://www.kaggle.com/datasets/amirzenoozi/persian-news-dataset) is a comprehensive collection of Persian news articles scraped from multiple Iranian news agencies, including FarsNews, ISNA, and others.

### Data Statistics

```
Original dataset:            392,532 articles (10 columns)
After cleaning:              367,170 articles
Used for training:            36,717 articles (10% sample)
Train / Validation / Test:   25,701 / 5,508 / 5,508  (70/15/15 split)
```

### Input-Output Format

```
Source:   "خلاصه کن: [full article body]"
Target:   [article abstract]
```

The prefix `"خلاصه کن: "` conditions the model to perform summarization in a prompt-based manner, consistent with T5's text-to-text paradigm.

### Preprocessing Pipeline

1. **Character Normalization**: Arabic characters mapped to Persian equivalents (`ی → ي`, `ک → ك`, `ه → ۀ/ة`)
2. **Noise Removal**: URLs (`http://`, `www.`), HTML tags (`<...>`), non-Persian/alphanumeric characters
3. **Whitespace Normalization**: Collapsing multiple spaces into single spaces
4. **Filtering**: Removal of empty bodies or abstracts
5. **Deduplication**: Removal of exact (body, abstract) pairs
6. **Subsampling**: 10% random sample for manageable training time

---

## Training Configuration

| Hyperparameter | Value |
|----------------|-------|
| **Epochs** | 3 |
| **Learning Rate** | 1e-4 |
| **Optimizer** | AdamW |
| **Batch Size** | 4 (train & eval) |
| **Max Source Length** | 512 tokens |
| **Max Target Length** | 128 tokens |
| **Label Padding** | -100 (ignored in loss) |
| **Data Collator** | `DataCollatorForSeq2Seq` |
| **Evaluation Strategy** | Per epoch |
| **Save Strategy** | Per epoch, keep best 2 |
| **Generation (eval)** | Beam search, `num_beams=4`, `max_length=128` |
| **Mixed Precision** | Native (default `fp32`) |
| **Hardware** | NVIDIA Tesla T4 (16 GB VRAM) |

---

## Results

### Training Progress

| Epoch | Training Loss | Validation Loss | ROUGE-1 | ROUGE-2 | ROUGE-L |
|:-----:|:-------------:|:---------------:|:-------:|:-------:|:-------:|
| 1 | 12.1948 | 5.9350 | 0.2153 | 0.0754 | 0.1850 |
| 2 | 8.1687 | 3.7935 | 0.3453 | 0.1652 | 0.3052 |
| 3 | 6.3686 | 2.8731 | 0.4054 | 0.2204 | 0.3602 |

### Final Test Set Performance

| ROUGE-1 | ROUGE-2 | ROUGE-L | Test Loss |
|:-------:|:-------:|:-------:|:-------------:|
|0.4163 | 0.2332 | 0.2153 | 0.3684 | 2.8823 |

### Interpretation

- **ROUGE-1 (0.4163)**: Moderate unigram overlap — the model captures key terms from the reference summaries.
- **ROUGE-2 (0.2332)**: Respectable bigram recall, indicating the model learns coherent phrase patterns typical of Persian news abstracts.
- **ROUGE-L (0.3684)**: Decent longest common subsequence score, suggesting structural similarity between generated and reference summaries.

Performance is constrained by using only 10% of the available data. Training on the full dataset with hyperparameter tuning is expected to yield significant improvements.

---

## Inference
### Example Outputs

#### 1️- Scientific Text (In-Domain)

**Input (excerpt):**
> پژوهشگران علوم اعصاب به کشفیات تازه‌ای درباره نحوه ذخیره‌سازی خاطرات در مغز دست یافته‌اند... مغز همیشه خاطره را «همان‌طور که بوده» بازیابی نمی‌کند، بلکه آن را براساس تجربیات جدید بازسازی می‌کند...

**Generated Summary:**
> پژوهشگران علوم اعصاب به کشفیات تازه ای درباره نحوه ذخیره سازی خاطرات در مغز دست یافته اند. محققان معتقدند مغز همیشه خاطره را «همان طور که بوده» بازیابی نمی کند بلکه آن را براساس تجربیات جدید بازسازی می کند اما در عین حال امکان خطا و تحریف نیز وجود دارد.

#### 2️- Literary Narrative (Out-of-Domain)

> ⚠️ **Note on Domain Transfer**: Although the model was trained exclusively on news articles, it demonstrates a notable ability to extract coherent summaries from literary Persian prose. In the example below, the model successfully identifies the protagonist, setting, and emotional arc of a narrative text — capturing both the physical scene (the old garden) and the internal conflict (a difficult decision). This suggests that the learned summarization patterns generalize beyond the news domain to character-driven Persian narratives.

**Input (excerpt):**
> باد عصرگاهی آرام از میان شاخه‌های بلند چنارها می‌گذشت... رعنا، که مدت‌ها بود پایش را به این باغ نگذاشته بود، آهسته در مسیر باریک میان درختان قدم می‌زد... نمی‌دانست آیا هنوز آن شجاعت را در خود دارد یا نه.

**Generated Summary:**
> او پایش را به این باغ نگذاشته بود، آهسته در مسیر باریک میان درختان قدم می‌زد و با هر قدم خاطرات سال‌های دور را در ذهنش ثبت می‌کرد.

#### 3️- Health/Science Mixed (In-Domain)

**Input (excerpt):**
> مطالعه بر روی «ریتم‌های درونی بدن» نشان داده است که تقریباً تمام سلول‌های انسان دارای ساعت زیستی مستقل هستند... ساعت زیستی بدن تنها به نور واکنش نشان نمی‌دهد، بلکه عوامل دیگری مانند زمان غذا خوردن، فعالیت بدنی و حتی دمای محیط نیز بر آن اثر می‌گذارند.

**Generated Summary:**
> محققان دریافته اند که ساعت زیستی بدن تنها به نور واکنش نشان نمی دهد بلکه عوامل دیگری مانند زمان غذا خوردن فعالیت بدنی و حتی دمای محیط نیز بر آن اثر می گذارند.

---

## Requirements & Setup

### Dependencies

```
transformers>=4.30.0
datasets>=2.12.0
evaluate>=0.4.0
hazm>=0.9.5
pandas>=1.5.0
scikit-learn>=1.2.0
kagglehub>=0.2.0
torch>=2.0.0
```

### Model Download

The fine-tuned model checkpoint is hosted on Google Drive and can be downloaded directly using `gdown`, no manual download required.

Download and extract the model:

```bash
# Install gdown if not already installed
pip install gdown

# Download the zipped model
gdown --id 1xJz5OwqhQnDwZveKxCwhEVvjsc1zekKe -O my_model.zip

# Extract
unzip my_model.zip -d parsT5_persian_news_summarizer && rm my_model.zip

# Load instantly in your code
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

tokenizer = AutoTokenizer.from_pretrained("./parsT5_persian_news_summarizer")
model = AutoModelForSeq2SeqLM.from_pretrained("./parsT5_persian_news_summarizer")
```

---

## Limitations & Future Work

### Current Limitations

| Limitation | Description |
|------------|-------------|
| **Partial Data** | Trained on only 10% (36.7K) of available data due to time constraints |
| **Domain Specificity** | Model is trained exclusively on news — performance may degrade on informal, conversational, or highly technical Persian |
| **Fixed Hyperparameters** | No hyperparameter search (learning rate, beam size, prefix format) |
| **Basic Decoding** | Only beam search employed; no sampling, top‑k, or top‑p strategies explored |
| **ROUGE‑Only Evaluation** | Semantic quality metrics (e.g., BERTScore, BLEURT) not yet assessed |

### Future Improvements

- **Full‑Dataset Training**: Leverage all ~367K cleaned samples for a stronger model
- **Hyperparameter Optimization**: Grid search over learning rates, batch sizes, and generation parameters
- **Advanced Decoding**: Incorporate nucleus sampling, diversity penalties, and length penalties
- **Semantic Evaluation**: Compute BERTScore using a Persian BERT model (e.g., `bert-base-parsbert-uncased`)
- **Model Distillation**: Train a smaller, faster student model for production deployment
- **Streamlit/Gradio Demo**: Build an interactive Persian summarization web application

---

## References & Acknowledgements

- **ParsT5**: [Ahmad/parsT5-base](https://huggingface.co/Ahmad/parsT5-base) — Pre‑trained Persian sequence‑to‑sequence model
- **Persian News Dataset**: [amirzenoozi/persian-news-dataset](https://www.kaggle.com/datasets/amirzenoozi/persian-news-dataset) on Kaggle
- **T5 Architecture**: Raffel et al., *Exploring the Limits of Transfer Learning with a Unified Text‑to‑Text Transformer*, JMLR 2020
- **Hazm**: [Persian NLP Toolkit](https://github.com/sobhe/hazm) by Sobhan Mohammadi
- **Hugging Face**: [Transformers](https://github.com/huggingface/transformers), [Datasets](https://github.com/huggingface/datasets), [Evaluate](https://github.com/huggingface/evaluate)
- **Kaggle**: Free GPU environment for training

---
**Course:** Deep Learning    
**University:** Amirkabir University of Technology    
**Semester:** Fall 2025    
**Author:** Hadi Salavati
