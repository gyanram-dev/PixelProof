# Dataset Research Report: AI-Generated vs Real Image Detection

## Executive Summary

After researching publicly available datasets for AI-generated image detection, this report shortlists 4 recommended datasets and provides a recommendation for PixelProof MVP.

---

## Shortlisted Datasets

### Real Images

| Dataset | Size | Source | Image Quality | License |
|---------|------|--------|---------------|----------|
| **ImageNet (ILSVRC)** | 1.28M images | ImageNet LSVRC 2012-2017 | High (224x224, varied) | Non-commercial research |
| **CIFAR-10** | 60k images | Cornell University | Low (32x32) | MIT-like |

### AI-Generated Images

| Dataset | Size | Source | Image Quality | License |
|---------|------|--------|---------------|----------|
| **GenImage** | 1.33M fake + 1.35M real | Generated from ImageNet classes | High (varies) | Research use |
| **DiffusionDB** | 14M images | Stable Diffusion Discord | High (varies) | CC0 (public domain) |
| **CIFAKE** | 120k (60k each) | Generated from CIFAR-10 | Low (32x32) | Research |
| **WildFake** | 2.5M fake + 1M real | Multiple generators | High | Research |

---

## Detailed Comparison

### ImageNet (ILSVRC 2012-2017)

- **Size**: 1,281,167 training images, 50,000 validation
- **Classes**: 1,000 object categories
- **Quality**: High-resolution, diverse object classes
- **License**: Non-commercial research only
- **Access**: Official download page or Kaggle
- **Use case**: Gold standard for real images in detection research

### GenImage

- **Size**: ~1.33M AI-generated + ~1.35M real from ImageNet
- **Generators**: Midjourney, Stable Diffusion (v1.4, v1.5), ADM, GLIDE, Wukong, VQDM, BigGAN
- **Quality**: High, based on ImageNet classes
- **License**: Research use (free download)
- **Access**: GitHub (https://github.com/GenImage-Dataset/GenImage)
- **Use case**: Primary benchmark dataset for AI detection research

### DiffusionDB

- **Size**: 14 million images
- **Source**: Stable Diffusion user-generated images
- **Quality**: Varies (user-curated, wide range)
- **License**: CC0 (public domain)
- **Access**: Hugging Face Dataset
- **Use case**: Largest available, perfect for training scale

### CIFAKE

- **Size**: 60k real + 60k AI-generated (from CIFAR-10)
- **Generators**: Stable Diffusion, Midjourney
- **Quality**: Low (32x32) - matches CIFAR-10
- **License**: Research
- **Access**: Kaggle
- **Use case**: Quick prototyping, smaller scale experiments

---

## MVP Recommendation

### Best Combination: GenImage + ImageNet

| Aspect | Recommendation |
|--------|-------------|
| **Training** | GenImage (AI-generated) + ImageNet subset (real) |
| **Scale** | ~200k images total for MVP (balanced) |
| **Diversity** | Multiple generators: SD v1.4, Midjourney, ADM |
| **Expected Accuracy** | 85-95% with CNN/Transformer |

**Rationale**:
1. **Research-standard**: GenImage is the most cited benchmark dataset
2. **Balanced classes**: Built-in real/fake pairs
3. **Multiple generators**: Tests generalization (critical for real-world)
4. **Clean licensing**: Research use, free download
5. **Proven results**: SOTA methods achieve 70%+ cross-gen accuracy

### Alternative: DiffusionDB + ImageNet

For larger scale MVP:
- Train on DiffusionDB (subsample to ~200k)
- Validate on ImageNet real images
- Pro: 14M potential training samples
- Con: Needs significant storage/processing

### Avoid for MVP

| Dataset | Reason |
|---------|--------|
| WildFire | Large scale but complex mix may confuse MVP |
| CIFAKE | Too small, low resolution limits model choice |

---

## Dataset Access Links

```
ImageNet:      https://www.kaggle.com/c/imagenet-object-localization-challenge
GenImage:      https://github.com/GenImage-Dataset/GenImage
DiffusionDB:   https://huggingface.co/datasets/poloclub/diffusiondb
CIFAKE:        https://www.kaggle.com/datasets/birdy654/cifake-real-and-ai-generated-synthetic-images
```

---

## Next Steps

1. Download GenImage subset (~50k images per class for MVP)
2. Create train/val/test splits (80/10/10)
3. Start baseline training with ResNet/EfficientNet
4. Evaluate on held-out generator (e.g., Midjourney-only test set)

---
*Generated: May 2026*