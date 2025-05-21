import hashlib
from typing import Any, Optional

import numpy as np
import torch
from ml_dtypes import bfloat16, float4_e2m1fn, float8_e4m3fn, float8_e5m2
from qdrant_client import QdrantClient
from qdrant_client.http.exceptions import UnexpectedResponse
from qdrant_client.http.models import Distance, PointStruct, VectorParams
from sentence_transformers import SentenceTransformer
from sentence_transformers.quantization import quantize_embeddings

from core.configs import ExperimentConfig, QuantizationType
from core.reduction import get_reducer
from core.reduction.base import DimensionalityReducer


class EmbeddingEngine:
    def __init__(
        self,
        model_name: str,
        quant_type: Optional[QuantizationType] = None,
        benchmark: Optional[str] = None,
        cache_location: str = ":memory:",
    ):
        
        device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
        model = SentenceTransformer(model_name, device=device, trust_remote_code=True)
        self.model: SentenceTransformer = torch.compile(model) if device in ["cuda", "cpu"] else model
        self.model_name = model_name
        self.device = device
        self.quant_type = quant_type
        self.benchmark = benchmark
        self.qdrant_client = QdrantClient(cache_location)
        self.calibration_dataset = None
        self.reduction_fit_dataset = None
        self.reducer: Optional[DimensionalityReducer] = None
        self.int8_ranges = None

        # attributes needed for MTEB
        self.model_card_data = self.model.model_card_data
        self.similarity_fn_name = self.model.similarity_fn_name

    def set_calibration_dataset(self, calibration_dataset: str) -> None:
        self.calibration_dataset = calibration_dataset

    def _get_calibration_embeddings(self) -> np.ndarray:
        if self.calibration_dataset:
            return self._return_all_embeddings(self.calibration_dataset)
        if not self.benchmark:
            raise ValueError("Benchmark is required for calibration dataset")
        return self._return_all_embeddings(self.benchmark)
    
    def set_reduction_fit_dataset(self, reduction_fit_dataset: str) -> None:
        """Set the dataset to use for fitting the dimensionality reduction method."""
        self.reduction_fit_dataset = reduction_fit_dataset

    def get_reduction_fit_embeddings(self) -> np.ndarray:
        """Get embeddings for the dataset used to fit the dimensionality reduction method."""
        
        if self.reduction_fit_dataset and self.collection_exists(self.reduction_fit_dataset):
            return self._return_all_embeddings(self.reduction_fit_dataset)

    def collection_exists(self, collection_name: str) -> bool:
        """Check if a Qdrant collection exists."""
        try:
            self.qdrant_client.get_collection(collection_name)
            return True
        except (UnexpectedResponse, ValueError):
            return False

    def _init_int8_ranges(self) -> None:
        """Initialize ranges for INT8 quantization using calibration embeddings"""
        calibration_embeddings = self._get_calibration_embeddings()
        
        # Apply reduction transformation if it exists
        if self.reducer:
            calibration_embeddings = self.reducer.transform(calibration_embeddings)

        # Calculate ranges (min, max) for each dimension
        self.int8_ranges = np.vstack(
            [
                np.min(calibration_embeddings, axis=0),
                np.max(calibration_embeddings, axis=0),
            ]
        )

    def set_reduction_config(self, experiment: ExperimentConfig) -> None:
        """Configure the reduction strategy according to an ExperimentConfig."""
        # Initialize appropriate reducer
        self.reducer = get_reducer(
            experiment,
            self.model_name,
            self.benchmark,
            self.device,
        )

    def fit_reduction(self) -> None:
        """Fit the dimensionality reduction method using the reduction_fit_dataset."""
        if not self.reducer:
            raise ValueError("Reducer not configured")
        embeddings = self.get_reduction_fit_embeddings()
        # Apply unified fit
        self.reducer.fit(embeddings)
        # Re-initialize INT8 ranges if needed
        if self.quant_type == QuantizationType.INT8:
            self._init_int8_ranges()

    def _return_all_embeddings(self, collection_name: str, limit=250_000) -> np.ndarray:
        """Return all embeddings from a collection."""
        embeddings = []
        offset = None
        batch_size = 1000  # Process 1000 vectors at a time
        total_vectors = 0

        while total_vectors < limit:
            # Adjust batch size for last iteration if needed
            current_batch_size = min(batch_size, limit - total_vectors)

            points, next_offset = self.qdrant_client.scroll(
                collection_name=collection_name,
                limit=current_batch_size,
                offset=offset,
                with_vectors=True,
            )

            if not points:
                break

            # Process this batch
            batch_embeddings = np.vstack([np.array(p.vector) for p in points])
            embeddings.append(batch_embeddings)
            total_vectors += len(points)

            # Update offset for next iteration
            if next_offset is None:
                break
            offset = next_offset

        return np.vstack(embeddings)

    def _create_collection(self, collection_name: str) -> None:
        """Create a Qdrant collection if it doesn't exist."""
        try:
            self.qdrant_client.get_collection(collection_name)
        except (UnexpectedResponse, ValueError):
            self.qdrant_client.create_collection(
                collection_name=collection_name,
                vectors_config=VectorParams(
                    size=self.model.get_sentence_embedding_dimension(),
                    distance=Distance.COSINE,
                    datatype="float32",
                ),
            )

    def set_quant_type(self, quant_type: QuantizationType) -> None:
        """Change the current quantization type being used."""
        self.quant_type = quant_type

    def set_benchmark(self, benchmark: str) -> None:
        """Change the current benchmark being used."""
        self.benchmark = benchmark
        self._create_collection(benchmark)

    def quantize_tensor(self, x: np.ndarray) -> np.ndarray:
        if self.quant_type is None:
            return x.astype(np.float32)

        if self.quant_type == QuantizationType.FLOAT32:
            return x.astype(np.float32)

        if self.quant_type == QuantizationType.FLOAT16:
            return x.astype(np.float16).astype(np.float32)

        elif self.quant_type == QuantizationType.BFLOAT16:
            return x.astype(bfloat16).astype(np.float32)

        elif self.quant_type == QuantizationType.FLOAT8_E4M3:
            return x.astype(float8_e4m3fn).astype(np.float32)

        elif self.quant_type == QuantizationType.FLOAT8_E5M2:
            return x.astype(float8_e5m2).astype(np.float32)

        elif self.quant_type == QuantizationType.FLOAT4_E2M1:
            return x.astype(float4_e2m1fn).astype(np.float32)

        elif self.quant_type == QuantizationType.INT8:



            if not self.benchmark:
                raise ValueError("Benchmark is required for INT8 quantization")
            
            calibration_embeddings = self._get_calibration_embeddings()
            # Apply reduction transformation if it exists
            if self.reducer:
                calibration_embeddings = self.reducer.transform(calibration_embeddings)
            return quantize_embeddings(
                x, precision="uint8", calibration_embeddings=calibration_embeddings
            ).astype(np.float32)

        elif self.quant_type == QuantizationType.BINARY:
            return (x > 0).astype(np.float32)

        return x

    def encode(self, sentences: list[str], **kwargs: Any) -> np.ndarray:
        # Calculate hashes for all sentences using MD5
        sentence_hashes = [
            int(hashlib.md5(s.encode()).hexdigest()[:16], 16)
            for s in sentences
        ]

        embeddings = np.zeros((len(sentences), self.model.get_sentence_embedding_dimension()))
        uncached_sentences = []
        uncached_indices = []

        # Try to get embeddings from cache in batches
        if self.benchmark:
            CACHE_BATCH = 1024
            for batch_start in range(0, len(sentences), CACHE_BATCH):
                batch_end = min(batch_start + CACHE_BATCH, len(sentences))
                batch_hashes = sentence_hashes[batch_start:batch_end]
                
                search_results = self.qdrant_client.retrieve(
                    collection_name=self.benchmark, ids=batch_hashes, with_vectors=True
                )

                # Create mapping of found embeddings
                found_embeddings = {r.id: np.array(r.vector) for r in search_results}

                # Process sentences in order
                for i, sentence_hash in enumerate(batch_hashes, start=batch_start):
                    if sentence_hash in found_embeddings:
                        embeddings[i] = found_embeddings[sentence_hash]
                    else:
                        uncached_sentences.append(sentences[i])
                        uncached_indices.append(i)
        else:
            uncached_sentences = sentences
            uncached_indices = list(range(len(sentences)))

        # Calculate embeddings for uncached sentences
        if uncached_sentences:
            with torch.no_grad():
                new_embeddings = self.model.encode(uncached_sentences, **kwargs)
                if isinstance(new_embeddings, torch.Tensor):
                    new_embeddings = new_embeddings.cpu().numpy()
                else:
                    new_embeddings = np.array(new_embeddings)

                # Store in cache if benchmark is set, in batches
                if self.benchmark:
                    UPLOAD_BATCH = 1024
                    for batch_start in range(0, len(uncached_sentences), UPLOAD_BATCH):
                        batch_end = min(batch_start + UPLOAD_BATCH, len(uncached_sentences))
                        batch_indices = uncached_indices[batch_start:batch_end]
                        batch_embeddings = new_embeddings[batch_start:batch_end]
                        
                        points = [
                            PointStruct(id=sentence_hashes[idx], vector=emb.tolist())
                            for idx, emb in zip(batch_indices, batch_embeddings)
                        ]
                        self.qdrant_client.upsert(
                            collection_name=self.benchmark, points=points
                        )

                # Place uncached embeddings in correct positions
                embeddings[uncached_indices] = new_embeddings

        # Apply dimension reduction if configured
        if self.reducer:
            embeddings = self.reducer.transform(embeddings)

        return self.quantize_tensor(embeddings)
