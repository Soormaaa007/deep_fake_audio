# 🧠 Audio Deepfake Detection — Model Documentation & Analysis

This project implements and compares three approaches for detecting fake audio samples:

- **Spectrogram-based CNN**
- **RawNet (1D CNN on raw waveform)**
- **Wav2Vec2 (Transformer-based pretrained model)**

---

## 🎯 Why These Models?

To cover a **diverse range of architectures**, I selected models that differ in input type, complexity, and learning approach:

| Model        | Why This Model? |
|--------------|-----------------|
| **CNN on Spectrograms** | Classic and effective for audio classification tasks. Spectrograms reveal key frequency differences in fake vs. real audio. Lightweight and fast to train. |
| **RawNet** (1D CNN on raw audio) | Processes raw waveform directly — no spectrogram needed. Captures fine-grained waveform details often missed in image-based models. Good for detecting subtle inconsistencies in fake voices. |
| **Wav2Vec2** | State-of-the-art model pretrained on massive speech data. Excellent for capturing high-level semantic and acoustic features. Represents the cutting edge of audio modeling. |

I intentionally avoided:

- **RNN-based models** (e.g., LSTM, GRU): Slower to train and often less accurate compared to CNNs on audio.
- **MFCC + Traditional ML**: Lower performance and outdated compared to deep models.
- **End-to-end GAN-based detection**: Complex to implement and overkill for a 5-day assessment with limited data.

This trio provides a strong balance between **simplicity, depth, and modernity**, giving a well-rounded evaluation of deepfake detection approaches.

---

## 🚧 Implementation Process

### ✅ Challenges Encountered

1. **Input Format Issues**  
   - CNN and RawNet required different input shapes (2D vs. 1D)
   - Wav2Vec2 expected raw waveforms with shape `[batch, time]`, which required reshaping

2. **Wav2Vec2 Shape Errors**  
   - Needed to remove extra dimensions using `.squeeze()` to avoid forward pass errors

3. **RawNet FC Layer Dimensions**  
   - Had to dynamically determine the flattened shape using a dummy input

4. **Underfitting in Wav2Vec2**  
   - Model performed poorly with frozen base layers and minimal fine-tuning

---

### 🛠️ Solutions Applied

- Standardized audio preprocessing
- Dynamically built model layers based on input
- Validated shapes and data flow through `print()` and `.shape` checks
- Used dummy inputs for RawNet FC layer
- Performed evaluation with accuracy & confusion matrix

---

### ⚙️ Assumptions Made

- Audio dataset has real/fake folder structure + key CSV file
- Audio duration is 1 second (16,000 samples)
- Dataset is balanced between real and fake samples

---

## 📊 Model Analysis

### 🧠 How Each Model Works

| Model            | Overview |
|------------------|----------|
| **CNN**          | Spectrogram → 2D CNN → FC layer |
| **RawNet**       | Raw waveform → 1D CNN → FC layer |
| **Wav2Vec2**     | Pretrained transformer → FC classifier |

---

### 📈 Performance Summary

| Model      | Accuracy | Notes                          |
|------------|----------|-------------------------------|
| CNN        | **92%**  | Strong baseline model         |
| RawNet     | 88%      | Efficient, but slightly weaker|
| Wav2Vec2   | 52%      | Underfit; needs fine-tuning   |

---

### ✅ Strengths & Weaknesses

| Model     | Strengths                        | Weaknesses                      |
|-----------|----------------------------------|----------------------------------|
| CNN       | Lightweight, fast, accurate     | Needs spectrogram conversion     |
| RawNet    | Raw input, no preprocessing     | Needs tuning, slightly lower acc |
| Wav2Vec2  | Powerful pretrained embeddings  | Needs longer training, GPU-heavy |

---

### 🚀 Suggestions for Improvement

- Fine-tune Wav2Vec2 with more epochs and unfrozen base layers
- Add data augmentation: noise, pitch shift, speed perturbation
- Expand dataset with more real-world fake voices
- Explore attention or hybrid CNN-RNN architectures

---

## 🧠 Reflection

### 1. What were the most significant challenges?

- Model input format mismatches
- Getting Wav2Vec2 to train properly
- Debugging shape and architecture issues

---

### 2. How might this approach perform in real-world conditions?

- Would require robust noise handling and domain adaptation
- Real-world deepfakes are more diverse and harder to catch

---

### 3. What additional data or resources would help?

- More diverse fake samples from different TTS models
- Multi-lingual or cross-speaker datasets
- Access to high-performance GPUs for longer training

---

### 4. How would you deploy this in production?

- Use CNN or RawNet for low-latency inference
- Deploy via FastAPI or Flask with TorchScript models
- Add audio pre-checks, confidence thresholding, and logging
- Monitor live data for drift and retrain periodically

---

## ✅ Submission Ready

- [x] Models trained and evaluated
- [x] Clear analysis with reflection
- [x] Ready for GitHub submission

---

## 📎 Instructions for Reproducibility

1. Clone the repo and install dependencies:  
   ```bash
   pip install -r requirements.txt
