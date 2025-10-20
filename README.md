# luna16-nodule-detection
CNN-based lung nodule detection on LUNA16 dataset using TensorFlow
A standardized CNN pipeline for detecting lung nodules in CT scans from the LUNA16 dataset.

## Setup
1. Clone: `git clone https://github.com//DLakshanMadushnka/luna16-nodule-detection.git`
2. Install: `pip install -r requirements.txt`
3. Configure paths in `config.py`.
4. Run: `python main.py`

## Results
- Test Accuracy: ~94%
- F1-Score: ~0.94 (balanced classes)
- See `outputs/` for models and plots.

## Usage
- Preprocessing: Balances positives/negatives with augmentation.
- Model: 3-layer CNN with batch norm and dropout.

## License
GNU
