# Image Caption Generator

A deep learning system that recognizes the context of an image and generates a natural
language caption describing it, combining a CNN for visual feature extraction with an
LSTM for language generation.

## How It Works

The pipeline follows the standard encoder-decoder architecture for image captioning:

1. **Feature extraction (encoder):** Each image is passed through **Xception**, a
   convolutional neural network pretrained on **ImageNet**, to extract a fixed-length
   feature vector capturing the image's visual content.
2. **Caption generation (decoder):** The extracted image features are fed into an
   **LSTM** network, which generates a sequence of English words one at a time,
   conditioned on both the image features and the words generated so far, producing a
   coherent caption.

```
Image → Xception (CNN, ImageNet-pretrained) → feature vector → LSTM → caption
```

## Requirements

```
tensorflow / keras
numpy
pillow
nltk  (for caption evaluation, if used)
```

## Usage

1. Extract image features using the pretrained Xception model.
2. Train the LSTM decoder on the extracted features paired with ground-truth captions.
3. Run inference on a new image to generate a caption.
