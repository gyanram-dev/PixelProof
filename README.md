# PixelProof

AI authenticity detection platform for classifying AI-generated vs real images.

## Project Structure

```
PixelProof/
├── app/              # FastAPI application
├── configs/          # Configuration files
├── data/             # Data directory
├── models/           # Model definitions
├── notebooks/        # Jupyter notebooks
├── pipelines/        # Training pipelines
├── artifacts/        # Saved models & outputs
└── requirements.txt # Python dependencies
```

## Setup

1. Create virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # Linux/Mac
   venv\Scripts\activate    # Windows
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Activate virtual environment:
   ```bash
   source venv/bin/activate  # Linux/Mac
   venv\Scripts\activate    # Windows
   ```

## Usage

```bash
# Run training pipeline
python pipelines/train.py

# Start API server
python -m uvicorn app.main:app --reload

# Run inference
python app/inference.py --image path/to/image.jpg
```

## License

MIT