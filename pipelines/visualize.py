import torch
import matplotlib.pyplot as plt
from pathlib import Path
import random

DATA_DIR = Path("data/processed")
NUM_SAMPLES = 5


def load_samples(label: str, num_samples: int = 5):
    """Load random tensor samples from a label directory."""
    label_dir = DATA_DIR / label
    
    if not label_dir.exists():
        print(f"Directory not found: {label_dir}")
        return [], []
    
    files = list(label_dir.glob("*.pt"))
    
    if not files:
        print(f"No .pt files found in {label_dir}")
        return [], []
    
    random.shuffle(files)
    files = files[:num_samples]
    
    tensors = []
    filenames = []
    
    for f in files:
        tensor = torch.load(f)
        tensors.append(tensor)
        filenames.append(f.stem)
    
    return tensors, filenames


def denormalize(tensor):
    """Reverse ImageNet normalization."""
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    
    denorm = tensor * std + mean
    denorm = torch.clamp(denorm, 0, 1)
    
    return denorm


def tensor_to_image(tensor):
    """Convert tensor to displayable numpy array."""
    if tensor.dim() == 4:
        tensor = tensor.squeeze(0)
    
    tensor = denormalize(tensor)
    
    img = tensor.permute(1, 2, 0).cpu().numpy()
    
    return img


def visualize_samples():
    """Visualize random samples from both classes."""
    
    print("Loading samples from processed dataset...")
    
    ai_tensors, ai_names = load_samples("ai_generated", NUM_SAMPLES)
    real_tensors, real_names = load_samples("real", NUM_SAMPLES)
    
    if not ai_tensors or not real_tensors:
        print("Error: No samples loaded. Run preprocessing pipeline first.")
        return
    
    fig, axes = plt.subplots(2, NUM_SAMPLES, figsize=(15, 6))
    
    for i in range(NUM_SAMPLES):
        if i < len(ai_tensors):
            img = tensor_to_image(ai_tensors[i])
            axes[0, i].imshow(img)
            axes[0, i].set_title(f"AI-Generated\n{ai_names[i][:20]}", fontsize=9)
            axes[0, i].axis("off")
        
        if i < len(real_tensors):
            img = tensor_to_image(real_tensors[i])
            axes[1, i].imshow(img)
            axes[1, i].set_title(f"Real\n{real_names[i][:20]}", fontsize=9)
            axes[1, i].axis("off")
    
    axes[0, 0].set_ylabel("AI-Generated", fontsize=12, rotation=0, labelpad=60)
    axes[1, 0].set_ylabel("Real", fontsize=12, rotation=0, labelpad=60)
    
    plt.suptitle("PixelProof Dataset Samples", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig("data/dataset_visualization.png", dpi=150, bbox_inches="tight")
    plt.show()
    
    print(f"\nVisualization saved to: data/dataset_visualization.png")
    
    print("\nDataset Info:")
    print(f"  Image size: {ai_tensors[0].shape}")
    print(f"  Num channels: {ai_tensors[0].shape[0]}")
    print(f"  Value range: [{ai_tensors[0].min():.3f}, {ai_tensors[0].max():.3f}]")
    print(f"  Normalization: ImageNet (mean=0.485,0.456,0.406 | std=0.229,0.224,0.225)")


def visualize_single(image_path: str):
    """Visualize a single processed image tensor."""
    tensor = torch.load(image_path)
    
    img = tensor_to_image(tensor)
    
    fig, ax = plt.subplots(1, 1, figsize=(6, 6))
    ax.imshow(img)
    ax.set_title(Path(image_path).stem, fontsize=12)
    ax.axis("off")
    plt.tight_layout()
    plt.show()


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Visualize dataset samples")
    parser.add_argument("--num-samples", type=int, default=5, help="Number of samples per class")
    parser.add_argument("--image", type=str, help="Single image path to visualize")
    args = parser.parse_args()
    
    if args.image:
        visualize_single(args.image)
    else:
        visualize_samples()


if __name__ == "__main__":
    main()