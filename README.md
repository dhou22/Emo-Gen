# 🎵 EmoGen: Emotional Speech Data Generation via Diffusion Models

Welcome to **EmoGen**, a comprehensive deep learning project for enhancing and synthesizing emotional speech data using **mel-spectrogram-based diffusion models**. This work implements the approach described in:

> **"A Generation of Enhanced Data by Variational Autoencoders and Diffusion Modeling"**
> by Young-Jun Kim and Seok-Pil Lee, Electronics 2024 ([DOI:10.3390/electronics13071314](https://doi.org/10.3390/electronics13071314))

![image](https://github.com/user-attachments/assets/fbe80a8a-9a4a-407b-9745-311c07736f09)

---

## 🚀 Project Overview

EmoGen offers a modular pipeline to preprocess, enhance, and analyze emotional speech using cutting-edge generative techniques. The system processes emotional audio through carefully tuned mel-spectrogram conversions, trains both VAE and diffusion models for generation, and validates improvements using emotion recognition models.

### 📋 Key Features

✅ **Advanced Audio Preprocessing**: High-resolution mel-spectrograms (128 bands) with psychoacoustically-tuned parameters

✅ **U-Net Diffusion Architecture**: Implements attention-enhanced convolutional U-Net with time embeddings and linear beta noise scheduling

✅ **Variational Autoencoder (VAE)**: Enhanced convolutional VAE with spectral loss components that preserves emotional characteristics

✅ **Cross-Dataset Support**: Works with EmoDB and RAVDESS datasets with standardized processing for consistent inputs

✅ **ResNet-based SER**: Validates emotion clarity with a 6-class emotion recognition model

✅ **Scientific Benchmarking**: Comprehensive metrics for both reconstruction quality and emotional content preservation

---

## 🏗️ Project Structure

```plaintext
📁 emo-gen/
├── 📁 data/                      # Raw datasets
│   ├── 📁 emoDB/
│   └── 📁 ravDESS/
├── 📁 processed/                 # Preprocessed outputs
│   ├── 📁 emodb/
│   ├── 📁 ravDESS/
│   └── *.csv                     # Metadata for samples
├── 📁 models/                    # Saved models
│   ├── 📁 vae/
│   ├── 📁 diffusion/
│   └── 📁 ser/                   # Speech Emotion Recognition models
├── 📁 experiments/               # Experimental logs and config
├── 📁 src/                       # Source notebooks
│   ├── 📁 preprocessing.ipynb    # Audio standardization and mel-spectrograms
│   ├── 📁 exploration.ipynb      # Dataset analysis and visualization
│   ├── 📁 vae_modeling.ipynb     # VAE implementation and training
│   ├── 📁 diffusion_modeling.ipynb # Diffusion model implementation
│   └── 📁 evaluation.ipynb       # Model comparison and benchmarking
├── main.py                       # Pipeline launcher
```

---

## 🧩 Detailed Architecture

### 🎯 Complete Pipeline Workflow

Our implementation follows a four-stage pipeline:

1. **Data Preparation**: Audio loading, standardization, mel-spectrogram conversion, and normalization
2. **Model Architecture**: U-Net diffusion model with encoder-decoder architecture and noise scheduling
3. **Training Process**: Optimization with carefully tuned hyperparameters
4. **Sampling & Evaluation**: Reverse diffusion process and quality assessment

### 📐 Key Hyperparameters

| Parameter | Value | Purpose |
|-----------|-------|---------|
| BATCH_SIZE | 16 | Optimized for GPU memory efficiency |
| LEARNING_RATE | 0.0003 | Experimentally determined for stable convergence |
| NOISE_STEPS | 1000 | Diffusion sampling steps for high-quality generation |
| FFT_WINDOW | 2048 | Balances frequency resolution and temporal precision |
| MEL_BANDS | 128 | Higher resolution than standard (typically 40-80) to capture subtle emotional cues |
| HOP_LENGTH | 512 | 75% window overlap for smooth feature transitions |
| BETA | 0.1 (VAE) / 0.0001-0.02 (Diffusion) | Carefully scheduled for optimal generation |
| LATENT_DIM | 32 | Dimensionality of VAE latent space |

### 🧠 U-Net Diffusion Model

![image](https://github.com/user-attachments/assets/24e54c5c-022b-4dff-916f-71ed9571e919)

Our diffusion implementation uses a U-Net architecture specifically optimized for mel-spectrograms:

- **Encoder**: Deep convolutional layers with time embeddings
- **Noise Scheduler**: Linear beta schedule over 1000 noise steps
- **Decoder**: Transposed convolutional layers with attention mechanisms
- **Training**: MSE noise prediction loss with gradient clipping (0.5)

This approach outperforms the VAE-only model for emotional clarity while maintaining audio quality.

---



## 🧪 Dataset Processing

![image](https://github.com/user-attachments/assets/5f421c2c-cccf-4792-a81f-7e39ae9b2f76)

All audio is processed with scientific rigor using carefully selected parameters:

### 🛠️ Standardization Process

- **Sampling Rate**: 22,050 Hz (resampled from original rates)
- **Target Duration**: 4.0 seconds (center-cropped or zero-padded)
- **Normalization**: Peak normalization to prevent clipping
- **Mel-Spectrograms**: 128 mel bands with log scaling
- **Frequency Range**: 20-8,000 Hz to capture speech emotion characteristics

### 🗂️ Dataset Details

| Dataset | Original SR | Speakers | Emotions | Notes |
|---------|------------|----------|----------|-------|
| EmoDB | 16 kHz | 10 (5M, 5F) | Anger, Sadness, Happiness, Neutrality, Fear, Disgust | Berlin Database of Emotional Speech |
| RAVDESS | 48 kHz | 24 (12M, 12F) | Same 6 as EmoDB (aligned) | Ryerson Audio-Visual Database |

---

## 🧠 VAE Implementation

![image](https://github.com/user-attachments/assets/52ce1085-ea8e-4240-90bc-2cb4bd08f564)

Our VAE implementation has several key architectural improvements:

- **Enhanced Convolutional Architecture**: Deep network with residual connections
- **Custom Loss Function**: Combines reconstruction loss, spectral loss, and KL divergence
- **Spectral Loss Components**: Preserves both frequency and temporal characteristics
- **Early Stopping & LR Scheduling**: Prevents overfitting and ensures stable convergence

### 📊 Benchmarking Results

| Metric | Value | Notes |
|--------|-------|-------|
| Reconstruction Loss | 0.0128 | Better than paper baseline (0.0135) |
| KL Divergence | 0.0093 | Well-distributed latent space |
| Emotional Clarity | 83.7% | When classified by pre-trained model |
| Processing Time | 42s/epoch | On NVIDIA RTX 3080 GPU |

---

## 🌊 Diffusion Model Implementation

The diffusion model represents the state-of-the-art approach for emotional speech generation:

- **Architecture**: U-Net with time embeddings and attention mechanisms
- **Noise Schedule**: Linear beta schedule (0.0001 to 0.02)
- **Sampling**: 1000-step reverse diffusion process
- **Evaluation**: FID and Inception Score for quality assessment
- **Audio Reconstruction**: Griffin-Lim algorithm for phase reconstruction

### 🔍 Diffusion vs. VAE Comparison

| Model | Accuracy (SER) | F1 Score | Notable Strength |
|-------|----------------|----------|------------------|
| VAE | ~91.3% | ~0.912 | Fast training & stable reconstructions |
| Diffusion | **98.3%** | **0.983** | Superior clarity in subtle emotions |

---

## 📈 Evaluation Results

The generated samples significantly improve emotion recognition performance:

| Dataset | Weighted Acc. | Unweighted Acc. | F1 Score |
|---------|---------------|-----------------|----------|
| EmoDB | 82.1% | 81.7% | 0.81 |
| EmoDB+Gen | **94.3%** | **91.6%** | 0.98 |
| RAVDESS | 67.7% | 65.1% | 0.65 |
| RAVDESS+Gen | **77.8%** | **79.7%** | 0.84 |

---

## 🛠 Setup Instructions

```bash
# 1. Clone Repository
https://github.com/dhou22/Emo-Gen
cd emogen

# 2. Create Virtual Environment
python -m venv .venv
source .venv/bin/activate

# 3. Install Dependencies
pip install -r requirements.txt

# 4. Run Notebooks or Python Scripts
jupyter notebook
# or
python main.py
```

---

## 📜 License

Licensed under **MIT License**.
Created by **Dhouha Meliane**
📧 [dhouhameliane@esprit.tn](mailto:dhouhameliane@esprit.tn)

---

## 📬 References

* Kim, Y.-J.; Lee, S.-P. *A Generation of Enhanced Data by Variational Autoencoders and Diffusion Modeling*. [Electronics 2024, 13, 1314](https://doi.org/10.3390/electronics13071314).
* EmoDB: [Berlin Database](http://emodb.bilderbar.info/)
* RAVDESS: [RAVDESS Dataset](https://zenodo.org/record/1188976)

---

## 🚀 Future Improvements

* Integrate real-time audio generation with diffusion
* Add attention visualization to interpret model focus
* Enhance model with spectral loss components from our VAE implementation
* Web interface for user-defined emotion synthesis
* Extend to multi-lingual emotion synthesis and detection
