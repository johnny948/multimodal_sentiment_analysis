"""
Fused sentiment analysis model using attention-based fusion for audio and vision modalities.
"""

import logging
import streamlit as st
from typing import Tuple, Optional, List
from PIL import Image
import torch
import torch.nn as nn
from pathlib import Path
import os

from .audio_model import predict_audio_sentiment, load_audio_model
from .vision_model import predict_vision_sentiment, load_vision_model
from ..utils.preprocessing import preprocess_audio_for_model, get_vision_transforms
from ..utils.sentiment_mapping import get_sentiment_mapping
from ..config.settings import AUDIO_MODEL_CONFIG

logger = logging.getLogger(__name__)

# Get project root directory
_current_file = Path(__file__).resolve()
PROJECT_ROOT = _current_file.parent.parent.parent


@st.cache_resource
def load_attention_fusion_model():
    """Load the attention-based fusion model"""
    try:
        # Define AttentionFusionModel class
        class AttentionFusionModel(nn.Module):
            """Attention-based fusion model for multimodal sentiment analysis"""
            
            def __init__(self, audio_dim, vision_dim, num_classes=3, hidden_dim=256, dropout=0.5):
                super(AttentionFusionModel, self).__init__()
                
                self.audio_dim = audio_dim
                self.vision_dim = vision_dim
                
                # Project both modalities to the same dimension for attention
                self.audio_proj = nn.Linear(audio_dim, hidden_dim)
                self.vision_proj = nn.Linear(vision_dim, hidden_dim)
                
                # Attention mechanism with Softmax for normalized modality weights
                self.attention = nn.Sequential(
                    nn.Linear(hidden_dim * 2, hidden_dim),
                    nn.Tanh(),
                    nn.Linear(hidden_dim, 2),
                    nn.Softmax(dim=-1)  # Normalized weights that sum to 1
                )
                
                # Classifier (input is hidden_dim, not hidden_dim * 2, because we use addition, not concatenation)
                self.classifier = nn.Sequential(
                    nn.Linear(hidden_dim, hidden_dim),  # Input is added features (256 dim)
                    nn.BatchNorm1d(hidden_dim),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                    nn.Linear(hidden_dim, hidden_dim // 2),
                    nn.BatchNorm1d(hidden_dim // 2),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                    nn.Linear(hidden_dim // 2, num_classes)
                )
            
            def forward(self, audio_features, vision_features):
                batch_size = audio_features.size(0)
                
                # Project to same dimension
                audio_proj = self.audio_proj(audio_features)
                vision_proj = self.vision_proj(vision_features)
                
                # Concatenate for attention computation
                concat_features = torch.cat([vision_proj, audio_proj], dim=1)
                
                # Compute attention weights
                attention_weights = self.attention(concat_features)
                
                # Apply attention weights
                vision_weighted = vision_proj * attention_weights[:, 0:1]
                audio_weighted = audio_proj * attention_weights[:, 1:2]
                
                # Fused features (addition, not concatenation - matches saved model)
                fused_features = vision_weighted + audio_weighted  # [batch, hidden_dim]
                
                # Classify
                logits = self.classifier(fused_features)
                
                return logits, attention_weights
        
        # Load model
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model_path = PROJECT_ROOT / "notebooks" / "best_attention_fusion_model.pth"
        
        if not model_path.exists():
            raise FileNotFoundError(f"Attention fusion model not found at {model_path}")
        
        # Model dimensions (from notebook)
        audio_dim = 256
        vision_dim = 2048
        num_classes = 3
        
        model = AttentionFusionModel(
            audio_dim=audio_dim,
            vision_dim=vision_dim,
            num_classes=num_classes,
            hidden_dim=256,
            dropout=0.5
        ).to(device)
        
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.eval()
        
        logger.info(f"Attention fusion model loaded from {model_path}")
        return model, device
        
    except Exception as e:
        logger.error(f"Failed to load attention fusion model: {e}")
        return None, None


def extract_audio_features_for_fusion(audio_bytes: bytes, device: torch.device):
    """Extract audio features for attention fusion"""
    try:
        # Load audio model
        audio_model, _, _, feature_extractor = load_audio_model()
        if audio_model is None:
            return None
        
        # Preprocess audio with feature extractor (same as notebook)
        import librosa
        import tempfile
        import os
        
        # Save audio bytes to temporary file
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_file:
            tmp_file.write(audio_bytes)
            tmp_file_path = tmp_file.name
        
        try:
            # Load and resample audio
            audio, sr = librosa.load(tmp_file_path, sr=16000)
            
            # Convert to tensor
            audio_tensor = torch.tensor(audio).float()
            max_length = int(5.0 * 16000)  # 5 seconds
            
            if len(audio_tensor) > max_length:
                audio_tensor = audio_tensor[:max_length]
            else:
                pad_len = max_length - len(audio_tensor)
                audio_tensor = torch.nn.functional.pad(audio_tensor, (0, pad_len))
            
            # Use feature extractor
            inputs = feature_extractor(
                audio_tensor.numpy(),
                sampling_rate=16000,
                max_length=max_length,
                truncation=True,
                padding="max_length",
                return_attention_mask=True,
                return_tensors="pt"
            )
            
            input_values = inputs.input_values.to(device)
            attention_mask = inputs.attention_mask.to(device)
            
            # Extract features (same as notebook)
            with torch.no_grad():
                audio_outputs = audio_model.wav2vec2(input_values, attention_mask=attention_mask)
                audio_features = audio_outputs.last_hidden_state.mean(dim=1)
                audio_features = audio_model.projector(audio_features)
            
            return audio_features
            
        finally:
            try:
                os.unlink(tmp_file_path)
            except:
                pass
        
    except Exception as e:
        logger.error(f"Error extracting audio features: {e}")
        return None


def extract_vision_features_for_fusion(image: Image.Image, device: torch.device):
    """Extract vision features for attention fusion (same as notebook)"""
    try:
        # Load vision model
        vision_model, _, _ = load_vision_model()
        if vision_model is None:
            return None
        
        # Extract ResNet features (before fc layer) - same as notebook
        def extract_resnet_features(model, img_tensor):
            x = model.conv1(img_tensor)
            x = model.bn1(x)
            x = model.relu(x)
            x = model.maxpool(x)
            x = model.layer1(x)
            x = model.layer2(x)
            x = model.layer3(x)
            x = model.layer4(x)
            x = model.avgpool(x)
            x = torch.flatten(x, 1)
            return x
        
        # Preprocess image (same transform as notebook: Resize, CenterCrop, ToTensor, Normalize)
        from torchvision import transforms
        vision_transform = transforms.Compose([
            transforms.Resize(224),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        
        image_tensor = vision_transform(image.convert('RGB')).unsqueeze(0).to(device)
        
        # Extract features
        with torch.no_grad():
            vision_features = extract_resnet_features(vision_model, image_tensor)
        
        return vision_features
        
    except Exception as e:
        logger.error(f"Error extracting vision features: {e}")
        return None


def predict_fused_sentiment(
    audio_bytes: Optional[bytes] = None,
    image: Optional[Image.Image] = None,
) -> Tuple[str, float]:
    """
    Implement attention-based fusion for audio and/or vision modalities.
    Missing modalities are set to zero features.

    Args:
        audio_bytes: Audio bytes for audio sentiment analysis
        image: Input image for vision sentiment analysis

    Returns:
        Tuple of (fused_sentiment, overall_confidence)
    """
    # Require at least one of audio or vision
    if not audio_bytes and not image:
        logger.error("Attention fusion requires at least audio or vision input")
        return "Error: Please provide at least audio or vision input", 0.0
    
    try:
        # Load attention fusion model
        fusion_model, device = load_attention_fusion_model()
        if fusion_model is None:
            logger.error("Attention fusion model not available")
            return "Error: Attention fusion model not available", 0.0
        
        # Extract features (set missing modality to zero)
        if audio_bytes:
            audio_features = extract_audio_features_for_fusion(audio_bytes, device)
            if audio_features is None:
                logger.warning("Audio feature extraction failed, using zero features")
                audio_features = torch.zeros(1, 256).to(device)  # audio_dim = 256
        else:
            # Missing audio: use zero features
            audio_features = torch.zeros(1, 256).to(device)
            logger.info("Audio not provided, using zero features for attention fusion")
        
        if image:
            vision_features = extract_vision_features_for_fusion(image, device)
            if vision_features is None:
                logger.warning("Vision feature extraction failed, using zero features")
                vision_features = torch.zeros(1, 2048).to(device)  # vision_dim = 2048
        else:
            # Missing vision: use zero features
            vision_features = torch.zeros(1, 2048).to(device)
            logger.info("Vision not provided, using zero features for attention fusion")
        
        # Run attention fusion (even with zero features for missing modalities)
        with torch.no_grad():
            outputs, attention_weights = fusion_model(audio_features, vision_features)
            probabilities = torch.softmax(outputs, dim=1)
            confidence, predicted = torch.max(probabilities, 1)
            
            sentiment_map = get_sentiment_mapping(3)
            sentiment = sentiment_map[predicted.item()]
            confidence_score = confidence.item()
            
            # Log attention weights
            avg_vision_weight = attention_weights[0, 0].item()
            avg_audio_weight = attention_weights[0, 1].item()
            logger.info(f"Attention weights - Vision: {avg_vision_weight:.4f}, Audio: {avg_audio_weight:.4f}")
            
            # Log which modalities were used
            if audio_bytes and image:
                logger.info("Using full attention fusion (Audio + Vision)")
            elif audio_bytes:
                logger.info("Using attention fusion (Audio only, Vision set to zero)")
            elif image:
                logger.info("Using attention fusion (Vision only, Audio set to zero)")
        
        logger.info(f"Attention-based fusion completed: {sentiment} (confidence: {confidence_score:.2f})")
        return sentiment, confidence_score
        
    except Exception as e:
        logger.error(f"Error in attention-based fusion: {e}")
        return f"Error: {str(e)}", 0.0


def get_fusion_strategy_info() -> dict:
    """Get information about the fusion strategy."""
    return {
        "strategy_name": "Attention-Based Fusion",
        "description": "Uses attention-based fusion for audio and/or vision modalities",
        "primary_method": "Attention-based fusion with learnable modality weights",
        "modality_weights": {"Audio": 0.5, "Vision": 0.5},
        "advantages": [
            "Adaptive attention weights learned from data",
            "Feature-level fusion for better representation",
            "Handles missing modalities by setting them to zero",
            "Real-time multi-modal prediction",
        ],
        "use_cases": [
            "Multi-modal content analysis (Audio + Vision)",
            "Enhanced sentiment accuracy with attention mechanism",
            "Single modality prediction with zero-padding",
            "Comprehensive emotional understanding",
        ],
    }


def analyze_modality_agreement(
    audio_bytes: Optional[bytes] = None,
    image: Optional[Image.Image] = None,
) -> dict:
    """
    Analyze agreement between different modalities.

    Args:
        audio_bytes: Audio bytes
        image: Input image

    Returns:
        Dictionary containing agreement analysis
    """
    results = {}

    if audio_bytes:
        audio_sentiment, audio_conf = predict_audio_sentiment(audio_bytes)
        results["audio"] = {"sentiment": audio_sentiment, "confidence": audio_conf}

    if image:
        vision_sentiment, vision_conf = predict_vision_sentiment(image)
        results["vision"] = {"sentiment": vision_sentiment, "confidence": vision_conf}

    if len(results) < 2:
        return {"agreement_level": "insufficient_modalities", "details": results}

    # Analyze agreement
    sentiments = [result["sentiment"] for result in results.values()]
    unique_sentiments = set(sentiments)

    if len(unique_sentiments) == 1:
        agreement_level = "perfect"
        agreement_score = 1.0
    elif len(unique_sentiments) == 2:
        agreement_level = "partial"
        agreement_score = 0.5
    else:
        agreement_level = "low"
        agreement_score = 0.0

    # Calculate confidence consistency
    confidences = [result["confidence"] for result in results.values()]
    confidence_std = sum(confidences) / len(confidences) if confidences else 0

    return {
        "agreement_level": agreement_level,
        "agreement_score": agreement_score,
        "modalities_analyzed": len(results),
        "sentiment_distribution": {s: sentiments.count(s) for s in unique_sentiments},
        "confidence_consistency": confidence_std,
        "individual_results": results,
        "recommendation": _get_agreement_recommendation(agreement_level, len(results)),
    }


def _get_agreement_recommendation(agreement_level: str, num_modalities: int) -> str:
    """Get recommendation based on agreement level."""
    if agreement_level == "perfect":
        return "High confidence in prediction - all modalities agree"
    elif agreement_level == "partial":
        return "Moderate confidence - consider modality-specific factors"
    elif agreement_level == "low":
        return "Low confidence - modalities disagree, consider context"
    else:
        return "Insufficient data for reliable fusion"
