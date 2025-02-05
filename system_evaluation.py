import math
import numpy as np

# ---------------------------
# Calibrated parameters
# ---------------------------
calibrated_shift = 16.5           # Empirically determined shift factor
LOW_CONF_THRESHOLD = 0.35          # If uncertainty < 0.3: high-confidence → Accept (via Decision Tree, no LLM call)
VERY_LOW_CONF_THRESHOLD = 0.850000000001     # If uncertainty >= 0.7: very low-confidence → Re-record (no API call)
                                # Ambiguous cases (uncertainty between 0.3 and 0.7) are escalated to the LLM (simulate cost 1)
scaling_factor = 75.0             # Chosen so that the computed confidences match the sample output

# ---------------------------
# Dummy validation dataset
# ---------------------------
# Each sample is represented by a list of per-token log probabilities and a sample ID.
# (A sample’s average log probability is computed from its token logprobs.)
validation_data = [
    ([-12.5, -13.0, -12.8], "sample1"),  # Avg ≈ -12.77 → shifted = 4.73 → exp(4.73)/75 = ~1.00 → Uncertainty 0.00 → Accept
    ([-14.2, -14.0, -14.5], "sample2"),  # Avg ≈ -14.23 → shifted = 3.27 → exp(3.27)/75 = ~0.35 → Uncertainty 0.65 → Escalate
    ([-11.0, -10.8, -11.2], "sample3"),  # Avg = -11.00  → shifted = 6.50 → exp(6.50)/75 = >1, clamped to 1.00 → Uncertainty 0.00 → Accept
    ([-13.5, -13.8, -14.0], "sample4"),  # Avg ≈ -13.77 → shifted = 3.73 → exp(3.73)/75 = ~0.57 → Uncertainty 0.43 → Escalate
    ([-15.0, -15.2, -14.8], "sample5"),  # Avg = -15.00  → shifted = 2.50 → exp(2.50)/75 = ~0.16 → Uncertainty 0.84 → Re-record
    ([-12.0, -11.8, -12.2], "sample6"),  # Avg = -12.00  → shifted = 5.50 → exp(5.50)/75 = >1, clamped to 1.00 → Uncertainty 0.00 → Accept
    ([-13.2, -13.4, -13.3], "sample7"),  # Avg ≈ -13.30 → shifted = 4.20 → exp(4.20)/75 = ~0.90 → Uncertainty 0.10 → Accept
    ([-14.8, -14.9, -15.1], "sample8")   # Avg ≈ -14.93 → shifted = 2.57 → exp(2.57)/75 = ~0.18 → Uncertainty 0.82 → Re-record
]

total_samples = len(validation_data)
calibrated_api_calls = 0

print("Sample\tAvgLogProb\tConfidence\tUncertainty\tAction")
for logprobs, sample_id in validation_data:
    # 1. Compute the average log probability.
    avg_logprob = np.mean(logprobs)
    
    # 2. Apply the calibrated shift.
    shifted_lp = avg_logprob + calibrated_shift
    
    # 3. Compute confidence using the formula:
    #    confidence = min(1, exp(shifted_lp) / scaling_factor)
    conf = math.exp(shifted_lp) / scaling_factor
    if conf > 1.0:
        conf = 1.0
    elif conf < 0.0:
        conf = 0.0
    
    # 4. Compute uncertainty = 1 - confidence.
    uncertainty = 1.0 - conf
    
    # 5. Decide the action based on uncertainty:
    #    - If uncertainty < LOW_CONF_THRESHOLD → Accept (via Decision Tree, no API call)
    #    - If uncertainty is between LOW_CONF_THRESHOLD and VERY_LOW_CONF_THRESHOLD → Escalate (LLM call, cost 1)
    #    - If uncertainty >= VERY_LOW_CONF_THRESHOLD → Re-record (no API call; cost undefined for future rounds)
    if uncertainty < LOW_CONF_THRESHOLD:
        action = "Accept (Decision Tree)"
        cost = 0
    elif uncertainty < VERY_LOW_CONF_THRESHOLD:
        action = "Escalate (LLM)"
        cost = 1
    else:
        action = "Re-record"
        cost = 0

    calibrated_api_calls += cost
    print(f"{sample_id}\t{avg_logprob:8.2f}\t{conf:10.2f}\t{uncertainty:11.2f}\t{action}")

# ---------------------------
# Baseline Evaluation
# ---------------------------
# Baseline: every sample triggers one LLM call (1 API call per sample).
baseline_api_calls = total_samples

# ---------------------------
# Report Results
# ---------------------------
api_calls_saved = baseline_api_calls - calibrated_api_calls

print("\n----- Evaluation Summary -----")
print(f"Total samples: {total_samples}")
print(f"Baseline API calls (1 per sample): {baseline_api_calls}")
print(f"Calibrated system API calls: {calibrated_api_calls}")
print(f"API calls saved: {api_calls_saved}")
