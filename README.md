ASL Sign Language Recognition — Clean UI

A university project: deep-learning based sign language recognizer (letters + words) with a clean Streamlit GUI.
The system accepts images or video frames and outputs the predicted letter or word. It combines CNNs and Transformer components, includes preprocessing, training loops (train / valid / test), simple camera capture, and text-to-speech for spoken output.

🔎 Project Overview

This project is a Deep Learning pipeline for recognizing American Sign Language (ASL) from images or video frames.
It contains two main recognition flows:

Letters: a CNN-based classifier (ResNet18) trained on ASL alphabet images.

Words: a Transformer-based model + preprocessing that handles word-level signs (frames extracted from videos or static images).

A polished Streamlit GUI ties everything together: upload or capture an image, choose letters/words mode, run prediction, and optionally play the predicted word/letter with TTS.

✨ Key Features

Two separate models:

CNN (ResNet18) for letters (fast, compact).

Transformer-based model for words (sequence modeling on frames).

Robust preprocessing and augmentation for better generalization.

Full training loop with train / validation / test splits.

Streamlit GUI:

Upload images or capture a snapshot from camera

Choose Letters / Words mode

Display prediction and play audio (TTS)

Clean, production-like UI (white, minimal)

Automatic class label loading from dataset folders (ImageFolder format).

Designed to reduce overfitting via augmentation and proper validation.

📁 Project Structure (recommended)
project/
├── models/
│   ├── letters_model.pth
│   └── words_model.pth
├── train/
│   ├── train_letters.py
│   └── train_words.py
├── utils/
│   ├── model_loader.py
│   └── preprocess.py
├── app.py               # single-file Streamlit app (or asl_project.py)
├── requirements.txt
└── README.md

🧩 Team & Responsibilities (محدث)

Team (project contributors and roles):

Mohmmed Emad Elraw — responsible for preprocessing for the letters CNN model (augmentation, normalization, resizing).

Zeyad Ahraf — implemented the letters CNN model (design, training strategy, augmentation).

Sarah Mahrous Mohamed — implemented the training loops (train / valid / test) and orchestrated model training for model stability.

Youssif Hisham Baiomy — contributed to camera integration and GUI support.

Omar Abdelhameed — created the GUI and integrated model inference with Streamlit (frontend logic, upload + camera UI).

Ahd Said Atia — responsible for project integration (gluing modules, automated loading of classes/models, readme & final packaging).

Hana Alaa Abderhman — implemented Transformer model code, preprocessing pipeline, and training scripts for word recognition.

Shahd Ahmed Khaled — responsible for Transformer testing and validation (evaluation metrics, confusion matrices).


🛠️ Requirements

Install dependencies:

pip install -r requirements.txt


Example requirements.txt:

streamlit
torch
torchvision
tqdm
pillow
pyttsx3
numpy
opencv-python


Adjust versions as needed for your environment. Use CUDA-enabled torch if GPU training is available.

🧭 How to use (quick start)

Prepare dataset folders in ImageFolder format:

dataset/
  train/
    classA/
    classB/
    ...
  test/
    classA/
    classB/
    ...


Letters dataset: ASL alphabet (folder per letter).

Words dataset: folder per word (images or frames extracted from videos).

Train models (if you don't have pre-trained .pth files):

python train/train_letters.py    # trains letters CNN (saves models/letters_model.pth)
python train/train_words.py      # trains words Transformer (saves models/words_model.pth)


Run the GUI:

streamlit run app.py


Choose mode (Letters / Words)

Upload an image or use camera snapshot

Click Predict to see result

For words, TTS will attempt to speak the predicted label

🔬 Training Details (recommended practices)
Letters model:

Architecture: ResNet18 (pretrained on ImageNet, final layer adjusted).

Loss: CrossEntropyLoss.

Optimizer: Adam with lr ≈ 1e-4.

Augmentation: RandomHorizontalFlip, RandomRotation, ColorJitter, Resize to 224×224, normalization with ImageNet mean/std.

Splits: Internal 90/10 split from train folder for train/val if no separate test exists; keep a held-out test folder if possible.

Words model:

Preprocess videos into frames (or use contiguous sequences of frames).

Use a lightweight Vision-Transformer or Transformer encoder (or CNN+Transformer hybrid) to model temporal patterns.

Optionally use frame-level augmentation and temporal jittering.

Evaluation: top-1 accuracy, confusion matrix, per-class precision/recall.

Early stopping / checkpointing:
Save best model by validation accuracy to models/*.pth.
Log training/validation losses for review.

✅ Evaluation & Metrics

Report:

Train / Validation / Test accuracy

Per-class precision & recall (important for imbalanced classes)

Confusion matrix visualization for final model

For transformer word model, consider sequence-level accuracy and per-frame agreement

🧾 Notes & Tips

Dataset structure: torchvision.datasets.ImageFolder expects a directory per class — used throughout the code to auto-load label names.

Reducing overfitting: heavy augmentation, dropout (if applicable in Transformer heads), and weight decay help improve generalization.

Hardware: training on CPU will be slow. Use a GPU-enabled environment (CUDA-capable PyTorch) for reasonable training times.

TTS: pyttsx3 works locally for speech; on some systems additional backends or adjustments may be needed.

Camera: Streamlit's st.camera_input() provides a quick snapshot capture — not a continuous live feed.

🧩 Files to check / edit before running

app.py — edit model paths if necessary: letters_model.pth, words_model.pth (expect under models/).

train/train_letters.py — set TRAIN_DIR to your letters folder path.

train/train_words.py — set TRAIN_DIR and TEST_DIR to your words folders.

📚 References & Data

ASL Alphabet dataset (Kaggle / public datasets) — used for letters training.

Your project-specific words dataset (frames extracted from videos) — used for words training.





Thanks to the entire team for their components and integration work:
Mohmmed, Zeyad, Sarah, Youssif, Omar, Ahd, Hana, Shahd — great teamwork!
