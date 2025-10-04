"""
CLIP Image Embedding Service

Uses CLIP (Contrastive Language-Image Pre-training) for image similarity search.
Much faster and more accurate than text-based matching.
"""

from typing import List, Optional
import numpy as np
from PIL import Image
from sentence_transformers import SentenceTransformer
import torch


class CLIPService:
    """Service for encoding images using CLIP model"""

    def __init__(self, model_name: str = "clip-ViT-B-32"):
        """Initialize CLIP model

        Args:
            model_name: CLIP model to use. Options:
                - clip-ViT-B-32 (default, 350MB, good balance)
                - clip-ViT-L-14 (larger, more accurate, slower)
                - clip-ViT-B-16 (similar to B-32)
        """
        print(f"🔄 Loading CLIP model: {model_name}...")

        # Check if CUDA is available
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"🖥️  Using device: {self.device}")

        # Load model
        self.model = SentenceTransformer(model_name, device=self.device)

        print(f"✅ CLIP model loaded successfully!")

    def encode_image(self, image_path: str) -> np.ndarray:
        """Encode a single image to embedding vector

        Args:
            image_path: Path to image file

        Returns:
            numpy array of shape (512,) containing the embedding
        """
        try:
            # Load and encode image
            image = Image.open(image_path).convert("RGB")
            embedding = self.model.encode(image, convert_to_numpy=True)

            return embedding

        except Exception as e:
            print(f"❌ Error encoding image {image_path}: {e}")
            raise

    def encode_images_batch(
        self, image_paths: List[str], batch_size: int = 8
    ) -> List[np.ndarray]:
        """Encode multiple images in batches (faster)

        Args:
            image_paths: List of image file paths
            batch_size: Number of images to process at once

        Returns:
            List of numpy arrays, each of shape (512,)
        """
        embeddings = []

        for i in range(0, len(image_paths), batch_size):
            batch_paths = image_paths[i : i + batch_size]

            # Load images
            images = []
            for path in batch_paths:
                try:
                    img = Image.open(path).convert("RGB")
                    images.append(img)
                except Exception as e:
                    print(f"⚠️  Error loading {path}: {e}")
                    # Use a blank image as placeholder
                    images.append(Image.new("RGB", (224, 224)))

            # Encode batch
            batch_embeddings = self.model.encode(
                images, convert_to_numpy=True, batch_size=len(images)
            )
            embeddings.extend(batch_embeddings)

            print(
                f"📊 Encoded {min(i + batch_size, len(image_paths))}/{len(image_paths)} images"
            )

        return embeddings

    @staticmethod
    def cosine_similarity(embedding1: np.ndarray, embedding2: np.ndarray) -> float:
        """Calculate cosine similarity between two embeddings

        Args:
            embedding1: First embedding vector
            embedding2: Second embedding vector

        Returns:
            Similarity score between -1 and 1 (higher is more similar)
        """
        # Normalize vectors
        embedding1_norm = embedding1 / (np.linalg.norm(embedding1) + 1e-8)
        embedding2_norm = embedding2 / (np.linalg.norm(embedding2) + 1e-8)

        # Calculate dot product (cosine similarity for normalized vectors)
        similarity = np.dot(embedding1_norm, embedding2_norm)

        return float(similarity)

    @staticmethod
    def cosine_similarity_batch(
        query_embedding: np.ndarray, embeddings: np.ndarray
    ) -> np.ndarray:
        """Calculate cosine similarity between query and multiple embeddings (vectorized)

        Args:
            query_embedding: Query embedding vector of shape (512,)
            embeddings: Array of embeddings of shape (n, 512)

        Returns:
            Array of similarity scores of shape (n,)
        """
        # Normalize query
        query_norm = query_embedding / (np.linalg.norm(query_embedding) + 1e-8)

        # Normalize all embeddings
        embeddings_norm = embeddings / (
            np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-8
        )

        # Calculate all similarities at once (much faster)
        similarities = np.dot(embeddings_norm, query_norm)

        return similarities


# Global instance (loaded once on startup)
_clip_service: Optional[CLIPService] = None


def get_clip_service() -> CLIPService:
    """Get or create global CLIP service instance"""
    global _clip_service

    if _clip_service is None:
        _clip_service = CLIPService()

    return _clip_service
