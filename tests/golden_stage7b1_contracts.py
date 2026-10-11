"""Deterministic expected outcomes for Stage 7B-1 contract admission."""
GOLDEN_CASES = {
    "valid_synthetic": ("ACCEPTED", ()),
    "predicted_source": ("REJECTED", ("DERIVED_FROM_NON_OBSERVED_SOURCE",)),
    "unknown_confidence": ("REJECTED", ("CONFIDENCE_UNKNOWN",)),
    "missing_provenance": ("REJECTED", ("MISSING_CALIBRATION_ID",)),
    "unknown_schema": ("REJECTED", ("unsupported schema_version: 9.9",)),
    "frame_time_mismatch": ("REJECTED", ("FRAME_TIME_MISMATCH",)),
}
