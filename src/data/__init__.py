"""Data ingestion and dataset abstractions package."""

from src.data.base_dataset import BasePolypDataset
from src.data.kvasir_dataset import KvasirSEGDataset

__all__ = ["BasePolypDataset", "KvasirSEGDataset"]
