# shift_calibration.py
import math
import numpy as np
import logging

# Set up basic logging configuration
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def compute_confidence_from_logprobs(logprobs, shift=18):
    """
    Convert a list of per-token log probabilities into a single 0..1 confidence.
    The shift value is added to the aggregated log probability before converting.
    
    :param logprobs: List of log probabilities.
    :param method: How to aggregate ('avg', 'median', or 'min').
    :param shift: The value to add to the aggregated log probability.
    :return: Confidence score (0..1).
    """
    if not logprobs:
        return 0.0

    
    lp = np.mean(logprobs)

    # Handle potential -infinity cases
    if math.isinf(lp):
        return 0.0

    shifted_lp = lp + shift
    # The division factor (10) is a scaling parameter that you may adjust.
    confidence = min(1.0, max(0.0, math.exp(shifted_lp) / 10))
    return confidence


def calibrate_confidence_shift(dataset, shift_range=np.arange(10, 30, 0.5)):
    """
    Calibrate the optimal shift value for the confidence calculation.
    
    :param dataset: List of tuples (logprobs, y) where:
                      - logprobs: List of per-token log probabilities from an audio sample.
                      - y: Binary label (1 if the transcription is correct, 0 otherwise).
    :param method: The aggregation method to use ('avg', 'median', or 'min').
    :param shift_range: Iterable of candidate shift values.
    :return: Tuple (best_shift, best_loss)
    """
    best_shift = None
    best_loss = float('inf')
    
    for shift in shift_range:
        errors = []
        for logprobs, y in dataset:
            pred_confidence = compute_confidence_from_logprobs(logprobs, shift=shift)
            errors.append((pred_confidence - y) ** 2)
        mse = np.mean(errors)
        logger.info(f"Shift candidate: {shift:.2f} -> MSE: {mse:.4f}")
        if mse < best_loss:
            best_loss = mse
            best_shift = shift
            
    return best_shift, best_loss


if __name__ == "__main__":
    # Each tuple is (list of logprobs, correctness label)
    dataset = [
         ([-12.5, -13.0, -12.8], 1),  # Likely correct transcription
         ([-14.2, -14.0, -14.5], 0),  # Likely incorrect transcription
         ([-11.0, -10.8, -11.2], 1),
         ([-15.0, -15.2, -14.8], 0),
    ]
    
    optimal_shift, mse = calibrate_confidence_shift(
        dataset, 
        shift_range=np.arange(10, 30, 0.5)
    )
    print(f"Optimal shift: {optimal_shift} with MSE: {mse:.4f}")
