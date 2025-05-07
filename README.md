# 🎵 EmoGen: Emotional Speech Data Generation via Diffusion Models

Welcome to **EmoGen**, a generative deep learning project for enhancing and synthesizing emotional speech data using **mel-spectrogram-based diffusion models**. This work is inspired by the paper:

> **"A Generation of Enhanced Data by Variational Autoencoders and Diffusion Modeling"**
> by Young-Jun Kim and Seok-Pil Lee, Electronics 2024 ([DOI:10.3390/electronics13071314](https://doi.org/10.3390/electronics13071314))

![image](https://github.com/user-attachments/assets/fbe80a8a-9a4a-407b-9745-311c07736f09)

---

## 🚀 Project Overview

EmoGen offers a modular pipeline to preprocess, enhance, and analyze emotional speech using cutting-edge generative techniques. The system augments datasets with clearer emotion signals, validated via emotion recognition models.

### 🧠 Key Features

✅ **Diffusion-based Audio Synthesis**: Utilizes stable diffusion on mel-spectrograms to generate high-fidelity emotional speech.

✅ **Variational Autoencoder (VAE) Support**: Implements a convolutional VAE model with class conditioning for comparison.

✅ **Emotion Embedding Modules**: Integrates utterance style and emotion into mel-spectrograms.

✅ **ResNet-based SER (Speech Emotion Recognition)**: Validates clarity improvement using a 6-class classification model.

✅ **Multi-dataset Support**: Works with EmoDB and RAVDESS, including sampling rate normalization and metadata alignment.

✅ **Jupyter Notebook Workflow**: Modular notebooks for preprocessing, exploration, modeling, and roadmap planning.

✅ **Reproducible Results**: All configurations and outputs are versioned and documented.

---

## 🏗️ Project Architecture

```plaintext
📁 emo-gen/
├── 📁 data/                        # Raw datasets
│   ├── 📁 emoDB/
│   └── 📁 ravDESS/
├── 📁 processed/                  # Preprocessed outputs
│   ├── 📁 emodb/
│   ├── 📁 ravDESS/
│   └── *.csv                   # Metadata for samples
├── 📁 models/                     # Saved SER and generation models
├── 📁 experiments/                # Experimental logs and config
├── 📁 src/                        # Source notebooks
│   ├── 📁 preprocessing.ipynb     # STFT, Mel conversion, normalization
│   ├── 📁 exploration.ipynb       # Visualizations, class distributions
│   ├── 📁 modeling.ipynb          # SER, VAE & Diffusion model training
│   └── 📁 roadmap.ipynb           # Development notes
├── main.py                     # Optional pipeline launcher
├── *.jpg / *.png               # Mel spectrogram examples
├── *.pdf                       # Research references
```

Developed using **PyCharm IDE**.

---

## 📊 Dataset Overview

### 🗂 EmoDB

* **Source**: Berlin Database of Emotional Speech
* **Sampling**: Originally 16kHz, resampled to 22.05kHz
* **Classes**: Anger, Sadness, Happiness, Neutrality, Fear, Disgust
* **Speakers**: 10 (5 male, 5 female)

### 🗂 RAVDESS

* **Source**: Ryerson Audio-Visual Database
* **Sampling**: Originally 48kHz, resampled to 22.05kHz
* **Classes Used**: Same 6 as EmoDB (for alignment)
* **Speakers**: 24 (12 male, 12 female)

All audio files were standardized to **10 seconds**, converted into **mel-spectrograms**, and normalized using **Z-score normalization**.

---

## 🧠 Modeling Comparison

EmoGen compares two powerful generative models for emotional audio synthesis:

### 📦 Variational Autoencoder (VAE)

* **Conditioned VAE** on emotion labels
* Encoder-decoder architecture using 2D convolutional blocks
* Reconstructs and generates emotion-specific mel-spectrograms
* **Loss Function**: Combination of MSE reconstruction loss and KL divergence (Beta-VAE)
* **Training Results**:

  * Clear reconstruction of input spectrograms
  * Latent space encodes emotion distributions
  * Best performance when β = 0.5 and latent dim = 128

### ✨ Diffusion Model

* Trained on mel-spectrograms with attention-enhanced ResNet blocks
* Utilizes emotion and utterance-style embeddings
* Better emotion clarity in spectrogram synthesis compared to VAE
* **Loss Function**: Optimized via reverse denoising objective (ELBO)

### 📊 Quantitative Comparison

| Model     | Accuracy (SER) | F1 Score  | Notable Strength                       |
| --------- | -------------- | --------- | -------------------------------------- |
| VAE       | \~91.3%        | \~0.912   | Fast training & stable reconstructions |
| Diffusion | **98.3%**      | **0.983** | Superior clarity in subtle emotions    |

Both models enhance the emotional content of generated audio, but **Diffusion Modeling** outperforms in terms of **emotion richness** and **classification accuracy**.

---

## 📈 Evaluation Results

| Dataset     | Weighted Acc. | Unweighted Acc. | F1 Score |
| ----------- | ------------- | --------------- | -------- |
| EmoDB       | 82.1%         | 81.7%           | 0.81     |
| EmoDB+Gen   | **94.3%**     | **91.6%**       | 0.98     |
| RAVDESS     | 67.7%         | 65.1%           | 0.65     |
| RAVDESS+Gen | **77.8%**     | **79.7%**       | 0.84     |

---

## 🛠 Setup Instructions

```bash
# 1. Clone Repository
https://github.com/your-username/emogen.git
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

## 🧪 Testing and Monitoring

* 📊 **Confusion Matrices** and per-class metrics available in `modeling.ipynb`
* 📉 Optionally extend to MLflow or TensorBoard for tracking
* 🔄 Add tests for data loaders, SER model predictions

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

* Integrate real-time audio generation with diffusion.
* Add attention visualization to interpret model focus.
* Web interface for user-defined emotion synthesis.
* Extend to multi-lingual emotion synthesis and detection.
