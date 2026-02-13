"""Prototype-based fallback for very low-shot training sessions."""

from __future__ import annotations

import time
from collections import defaultdict

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from app.ml.dataset import LandmarkDataset
from app.ml.model import SignTransformer
from app.ml.trainer import TrainingMetrics


def run_prototypical_fallback(
    *,
    model: SignTransformer,
    train_dataset: LandmarkDataset,
    val_dataset: LandmarkDataset,
    device: str = "cpu",
    batch_size: int = 64,
) -> tuple[TrainingMetrics, dict[int, torch.Tensor]]:
    """
    Adapt classifier weights from class prototypes when target class has very few samples.

    Steps:
    - Compute normalized embedding prototypes per class from training set.
    - Initialize classifier rows from prototypes.
    - Evaluate train/validation metrics using the adapted model.
    """
    started_at = time.time()
    torch_device = torch.device(device)
    model.to(torch_device)
    model.eval()

    prototypes = compute_class_prototypes(
        model=model,
        dataset=train_dataset,
        device=torch_device,
        batch_size=batch_size,
    )
    apply_prototypes_to_classifier(model=model, prototypes=prototypes, device=torch_device)

    train_loss, train_acc = evaluate_model(
        model=model,
        dataset=train_dataset,
        device=torch_device,
        batch_size=batch_size,
    )
    val_loss, val_acc = evaluate_model(
        model=model,
        dataset=val_dataset,
        device=torch_device,
        batch_size=batch_size,
    )

    metric = TrainingMetrics(
        epoch=1,
        train_loss=train_loss,
        train_accuracy=train_acc,
        val_loss=val_loss,
        val_accuracy=val_acc,
        learning_rate=0.0,
        duration_sec=time.time() - started_at,
    )
    return metric, prototypes


def compute_class_prototypes(
    *,
    model: SignTransformer,
    dataset: LandmarkDataset,
    device: torch.device,
    batch_size: int = 64,
) -> dict[int, torch.Tensor]:
    """Compute normalized embedding mean vectors by class label."""
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
    )

    sums: dict[int, torch.Tensor] = {}
    counts: dict[int, int] = defaultdict(int)

    with torch.no_grad():
        for landmarks, labels in loader:
            landmarks = landmarks.to(device)
            labels = labels.to(device)

            embeddings = model.extract_embeddings(landmarks)
            embeddings = F.normalize(embeddings, dim=1)

            for index in range(labels.size(0)):
                label = int(labels[index].item())
                vector = embeddings[index]
                if label not in sums:
                    sums[label] = vector.clone()
                else:
                    sums[label] += vector
                counts[label] += 1

    prototypes: dict[int, torch.Tensor] = {}
    for label, sum_vector in sums.items():
        denominator = max(1, counts[label])
        prototype = sum_vector / float(denominator)
        prototypes[label] = F.normalize(prototype, dim=0)
    return prototypes


def apply_prototypes_to_classifier(
    *,
    model: SignTransformer,
    prototypes: dict[int, torch.Tensor],
    device: torch.device,
) -> None:
    """Map prototypes to linear classifier weights (cosine-like logits)."""
    classifier = model.classifier
    with torch.no_grad():
        for label, prototype in prototypes.items():
            if label >= classifier.out_features:
                continue
            classifier.weight[label].copy_(prototype.to(device))
            if classifier.bias is not None:
                classifier.bias[label] = -0.5 * torch.dot(prototype, prototype)


def evaluate_model(
    *,
    model: SignTransformer,
    dataset: LandmarkDataset,
    device: torch.device,
    batch_size: int = 64,
) -> tuple[float, float]:
    """Evaluate loss/accuracy on a dataset."""
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
    )
    if len(loader) == 0:
        return 0.0, 0.0

    total_loss = 0.0
    total_correct = 0
    total_samples = 0
    criterion = torch.nn.CrossEntropyLoss()

    with torch.no_grad():
        for landmarks, labels in loader:
            landmarks = landmarks.to(device)
            labels = labels.to(device)

            logits = model(landmarks)
            loss = criterion(logits, labels)

            predictions = torch.argmax(logits, dim=1)
            total_correct += int((predictions == labels).sum().item())
            total_samples += int(labels.size(0))
            total_loss += float(loss.item())

    mean_loss = total_loss / len(loader)
    accuracy = float(total_correct / total_samples) if total_samples > 0 else 0.0
    return mean_loss, accuracy
