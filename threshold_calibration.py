# threshold_calibration.py

import numpy as np
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

def predict_action(uncertainty, low_threshold, high_threshold):
    """
    Given an uncertainty value and thresholds, return a predicted action.
    
    Actions:
      0: Accept        if uncertainty < low_threshold
      1: Escalate      if low_threshold <= uncertainty < high_threshold
      2: Re-record     if uncertainty >= high_threshold
    """
    if uncertainty < low_threshold:
        return 0  # Accept
    elif uncertainty < high_threshold:
        return 1  # Escalate
    else:
        return 2  # Re-record

def evaluate_thresholds(validation_data, low_threshold, high_threshold):
    """
    Evaluate the accuracy of a given threshold pair over the validation dataset.
    
    :param validation_data: list of tuples (uncertainty, label)
    :param low_threshold: candidate threshold for accepting responses
    :param high_threshold: candidate threshold for escalating responses
    :return: accuracy (fraction of samples where predicted action == label)
    """
    predictions = []
    for uncertainty, label in validation_data:
        pred = predict_action(uncertainty, low_threshold, high_threshold)
        predictions.append(pred)
    
    labels = [label for (_, label) in validation_data]
    accuracy = np.mean(np.array(predictions) == np.array(labels))
    return accuracy

def calibrate_thresholds(validation_data, 
                         low_range=np.arange(0.1, 0.5, 0.05), 
                         high_range=np.arange(0.5, 1.0, 0.05)):
    """
    Grid search for the best pair of thresholds (low and high) over the validation data.
    
    :param validation_data: list of tuples (uncertainty, label)
    :param low_range: iterable of candidate low_threshold values.
    :param high_range: iterable of candidate high_threshold values.
    :return: (best_low, best_high, best_accuracy)
    """
    best_low = None
    best_high = None
    best_acc = -1

    for low in low_range:
        for high in high_range:
            if low >= high:
                continue  # Make sure low_threshold is less than high_threshold
            acc = evaluate_thresholds(validation_data, low, high)
            logger.info(f"Testing low={low:.2f}, high={high:.2f}: accuracy={acc:.4f}")
            if acc > best_acc:
                best_acc = acc
                best_low = low
                best_high = high

    return best_low, best_high, best_acc

if __name__ == "__main__":
    # Dummy validation dataset for demonstration.
    # Format: (uncertainty, label)
    # For example, if you have:
    #   uncertainty < 0.3  => the transcription is clear (Accept -> label 0)
    #   uncertainty between 0.3 and 0.7 => ambiguous (Escalate -> label 1)
    #   uncertainty >= 0.7 => very ambiguous (Re-record -> label 2)
    validation_data = [
         (0.20, 0),  # Low uncertainty, should Accept.
         (0.25, 0),
         (0.35, 1),  # Mid uncertainty, should Escalate.
         (0.40, 1),
         (0.50, 1),
         (0.65, 1),
         (0.75, 2),  # High uncertainty, should Re-record.
         (0.80, 2),
         (0.85, 2),
         (0.90, 2),
    ]
    
    best_low, best_high, best_acc = calibrate_thresholds(validation_data)
    print(f"Optimal thresholds found:")
    print(f"  LOW_CONF_THRESHOLD = {best_low}")
    print(f"  VERY_LOW_CONF_THRESHOLD = {best_high}")
    print(f"  Accuracy: {best_acc:.4f}")
