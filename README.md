# Multimodal Sentiment Analysis

A comprehensive multimodal sentiment analysis system that combines fine-tuned Wav2Vec2 (audio) and ResNet-50 (vision) models using attention-based fusion. The project includes a production-ready Streamlit web application for real-time sentiment analysis across multiple input modalities.

**Note: Some training notebooks may not render correctly in GitHub preview due to widget metadata compatibility issues. Please download and open them locally using Jupyter Notebook if needed.**



## Overview

This project implements state-of-the-art multimodal sentiment analysis by:

- **Single-Modal Models**: Fine-tuned Wav2Vec2-base for audio and ResNet-50 for vision sentiment analysis
- **Multimodal Fusion**: Attention-based fusion mechanism with Softmax normalization for adaptive modality weighting
- **Production System**: Complete Streamlit web application with multiple input methods and real-time inference

## Key Features

### Models

1. **Audio Sentiment Analysis**
   - Model: Fine-tuned Wav2Vec2-base (`facebook/wav2vec2-base`)
   - Training: Two-stage fine-tuning on RAVDESS dataset
   - Performance: 79.17% accuracy on RAVDESS test set
   - Features: Automatic preprocessing (16kHz sampling, 5s max duration)

2. **Vision Sentiment Analysis**
   - Model: Fine-tuned ResNet-50 (ImageNet pre-trained)
   - Training: End-to-end fine-tuning on RAVDESS frames
   - Performance: 70.71% accuracy on RAVDESS test set
   - Features: Face detection, automatic cropping, grayscale conversion

3. **Attention-Based Fusion**
   - Architecture: Custom AttentionFusionModel with Softmax-normalized attention weights
   - Performance: 92.31% accuracy on RAVDESS test set (4.17% improvement over feature concatenation)
   - Features: Dynamic modality weighting, missing modality handling, interpretable attention weights

### Web Application

- **Multiple Input Methods**: File upload, camera capture, microphone recording
- **Real-time Inference**: Instant sentiment predictions with confidence scores
- **Attention Visualization**: Display attention weights for interpretability
- **Missing Modality Support**: Graceful degradation when one modality is unavailable
- **Multi-page Interface**: Clean navigation between different analysis modes

## Project Structure

```
multimodal-sentiment-analysis/
├── app.py                          # Main Streamlit application
├── requirements.txt                # Python dependencies
├── pyproject.toml                 # Project configuration
├── Dockerfile                     # Container deployment
├── README.md                      # This file
├── notebooks/                     # Training and evaluation notebooks
│   ├── audio_wav2vec2_training.ipynb           # Audio model training
│   ├── vision_resnet50_training.ipynb          # Vision model training
│   ├── late_fusion_attention_based.ipynb       # Attention fusion training
│   ├── late_fusion_feature_concatenation.ipynb # Concatenation fusion baseline
│   └── late_fusion_missing_modality_evaluation copy.ipynb  # Missing modality evaluation
├── model_weights/                 # Model storage directory
└── src/                           # Source code package
    ├── __init__.py
    ├── config/                    # Configuration settings
    │   ├── __init__.py
    │   └── settings.py
    ├── models/                     # Model inference code
    │   ├── __init__.py
    │   ├── audio_model.py          # Wav2Vec2 audio model
    │   ├── vision_model.py         # ResNet-50 vision model
    │   ├── fused_model.py          # Attention-based fusion model
    │   └── text_model.py           # Text sentiment model (TextBlob)
    ├── utils/                      # Utility functions
    │   ├── __init__.py
    │   ├── preprocessing.py        # Data preprocessing utilities
    │   ├── file_handling.py        # File I/O utilities
    │   ├── sentiment_mapping.py    # Sentiment label mapping
    │   └── simple_model_manager.py # Model loading and management
    └── ui/                         # UI components
        ├── __init__.py
        └── styles.py               # Custom CSS styles
```

## Installation

### Prerequisites

- Python 3.9 or higher
- 4GB+ RAM (for model loading)
- CUDA-capable GPU (optional, for faster inference)

### Setup

1. **Clone the repository**:
   ```bash
   git clone <repository-url>
   cd multimodal-sentiment-analysis
   ```

2. **Create a virtual environment**:
   ```bash
   python -m venv venv
   
   # On Windows
   venv\Scripts\activate
   
   # On macOS/Linux
   source venv/bin/activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Set up model files**:
   
   Place the trained model files in the following locations:
   - Audio model: `notebooks/best_wav2vec2_two_stage.pth`
   - Vision model: `model_weights/resnet50_fer2013_sentiment.pth` or `notebooks/resnet50_fer2013_sentiment.pth`
   - Fusion model: `notebooks/best_attention_fusion_model.pth`
   
   The application will automatically load models from these local paths. Ensure all model files are present before running the application.

## Usage

### Running the Web Application

1. **Start the Streamlit application**:
   ```bash
   streamlit run app.py
   ```

2. **Open your browser** and navigate to `http://localhost:8501`

3. **Navigate between pages** using the sidebar:
   - 🏠 **Home**: Overview and model information
   - 📝 **Text Sentiment**: Analyze text with TextBlob
   - 🎵 **Audio Sentiment**: Analyze audio files or record with microphone
   - 🖼️ **Vision Sentiment**: Analyze images or capture with camera
   - 🔗 **Fused Model**: Combine audio and vision models
   - 🎬 **Attention Based Fusion (Video)**: Video-based analysis with attention fusion

### Input Methods

- **File Upload**: Support for audio (WAV, MP3, M4A, FLAC), images (PNG, JPG, JPEG, BMP, TIFF), and videos (MP4, AVI, MOV, MKV, WMV, FLV)
- **Camera Capture**: Direct image capture for vision analysis
- **Microphone Recording**: Real-time audio recording (max 5 seconds)

## Model Architecture

### Attention-Based Fusion Model

The fusion model uses a Softmax-normalized attention mechanism to dynamically weight audio and vision features:

```python
# Project modalities to shared dimension
audio_proj = Linear(audio_dim, hidden_dim)  # 256 → 256
vision_proj = Linear(vision_dim, hidden_dim)  # 2048 → 256

# Compute attention weights
concat_features = Concat([vision_proj, audio_proj])  # [batch, 512]
attention_weights = Softmax(Linear(Tanh(Linear(concat_features))))  # [batch, 2]

# Weighted fusion
fused_features = vision_proj * attention_weights[:, 0] + audio_proj * attention_weights[:, 1]

# Classification
logits = MLP(fused_features)  # 3-layer MLP: 256 → 256 → 128 → 3
```

**Key Features**:
- Softmax normalization ensures attention weights sum to 1
- Handles missing modalities by setting features to zero
- Provides interpretable attention weights for each modality

## Training Details

### Audio Model (Wav2Vec2)

- **Pre-training**: `facebook/wav2vec2-base` (self-supervised on 960h of LibriSpeech)
- **Fine-tuning Strategy**: Two-stage approach
  - Stage 1: Freeze backbone, train classifier (5 epochs, lr=3e-4)
  - Stage 2: Unfreeze with layer-specific learning rates (15 epochs)
    - Feature extractor: 0.0 (frozen)
    - Encoder: 1e-5
    - Feature projection: 1e-5
    - Classifier: 5e-4
- **Data Augmentation**: Noise addition, random volume, time stretching
- **Regularization**: Weight decay (0.01), gradient clipping, class weighting

### Vision Model (ResNet-50)

- **Pre-training**: ImageNet-1K V2 weights
- **Fine-tuning Strategy**: End-to-end training (all layers trainable)
- **Data Augmentation**: RandomCrop, RandomHorizontalFlip, RandAugment, RandomErasing
- **Regularization**: Label smoothing (0.1), weight decay (1e-4), dropout, early stopping

### Fusion Model

- **Training**: Feature-level fusion with frozen single-modal models
- **Optimization**: AdamW optimizer, cosine annealing LR scheduler with warmup
- **Regularization**: Dropout (0.5), BatchNorm, weight decay
- **Evaluation**: Speaker-independent split (actors 1-18 train, 19-21 val, 22-24 test)

## Experimental Results

### Performance on RAVDESS Dataset

| Model | Accuracy | F1-Score (Macro) | Precision | Recall |
|-------|----------|------------------|-----------|--------|
| Audio (Wav2Vec2) | 79.17% | 0.7845 | 0.7891 | 0.7812 |
| Vision (ResNet-50) | 70.71% | 0.7012 | 0.7089 | 0.6934 |
| Feature Concatenation | 88.14% | 0.8541 | 0.8612 | 0.8473 |
| **Attention-Based Fusion** | **92.31%** | **0.9073** | **0.9124** | **0.9021** |

### Missing Modality Analysis

| Scenario | Accuracy | F1-Score |
|----------|----------|----------|
| Full Fusion (Audio + Vision) | 92.31% | 0.9073 |
| Audio-Only (Vision set to zero) | 68.31% | 0.6712 |
| Vision-Only (Audio set to zero) | 78.49% | 0.7745 |

## Technical Details

### Data Preprocessing

**Audio**:
- Resampling to 16kHz
- Max duration: 5 seconds
- Silence trimming (top_db=30)
- Feature extraction via AutoFeatureExtractor

**Vision**:
- Face detection using OpenCV Haar Cascade
- Face cropping with configurable padding
- Grayscale conversion
- Resize to 224×224
- ImageNet normalization

### Data Split Strategy

To prevent data leakage, the dataset is split by actor ID (speaker-independent):
- **Training**: Actors 1-18 (1828 samples)
- **Validation**: Actors 19-21 (312 samples)
- **Test**: Actors 22-24 (312 samples)

This ensures no actor appears in multiple splits, preventing the model from memorizing actor-specific characteristics.

## Dependencies

Key libraries:

- **Streamlit**: Web application framework
- **PyTorch**: Deep learning framework
- **Transformers**: Hugging Face model library (Wav2Vec2)
- **Torchvision**: Computer vision models (ResNet-50)
- **OpenCV**: Face detection and image processing
- **Librosa**: Audio processing and feature extraction
- **MoviePy**: Video processing and audio extraction
- **Pillow**: Image processing
- **NumPy/Pandas**: Data manipulation

See `requirements.txt` for the complete list.

## Deployment

### Docker Deployment

```bash
# Build the container
docker build -t multimodal-sentiment-analysis .

# Run the container
docker run -p 8501:8501 multimodal-sentiment-analysis
```

### Local Development

```bash
# Run with custom port
streamlit run app.py --server.port 8502

# Run with custom address
streamlit run app.py --server.address 0.0.0.0
```

## Troubleshooting

### Common Issues

1. **Model Loading Errors**:
   - Ensure model files are in the correct locations
   - Check file permissions
   - Verify sufficient RAM (4GB+ recommended)

2. **CUDA/GPU Issues**:
   - Install PyTorch with CUDA support: `pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118`
   - Models will fall back to CPU if CUDA is unavailable

3. **Dependency Conflicts**:
   - Use a virtual environment
   - Ensure Python version is 3.9+

4. **Face Detection Fails**:
   - Ensure OpenCV is properly installed: `pip install opencv-python-headless`
   - The system will use center crop as fallback

## Citation

If you use this code or models in your research, please cite:

```bibtex
@misc{multimodal-sentiment-analysis,
  title={Multimodal Sentiment Analysis with Attention-Based Fusion},
  author={Your Name},
  year={2024},
  howpublished={\url{https://github.com/yourusername/multimodal-sentiment-analysis}}
}
```

## Acknowledgments

We thank the creators of the RAVDESS dataset for providing a comprehensive multimodal emotion recognition benchmark. We also acknowledge the open-source community for providing pre-trained models (Wav2Vec2 and ResNet-50) and frameworks (PyTorch, Transformers, Streamlit) that enabled this research.

## Contact

For questions or issues, please open an issue on GitHub or contact [your-email@example.com].
