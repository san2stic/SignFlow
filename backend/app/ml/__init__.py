"""Machine learning modules for SignFlow inference and training.

Modules
-------
pipeline            Sliding-window landmark inference state machine (V1 / V2).
features            Feature extraction: ENRICHED_FEATURE_DIM=493 (V1), ENRICHED_FEATURE_DIM_V2=611 (V2).
handshape_features  Handshape geometry features (42 dims/hand, 84 total).
facial_action_units Non-manual marker features from facial landmarks (32 dims).
signing_space       LSFB signing space encoding (18 dims).
model               SignTransformer architecture (V1).
model_v2            SignTransformerV2 multi-stream architecture (~5M params, V2).
sign_segmentation   SignBoundaryDetector BiLSTM for sign boundary detection.
conversation_context ConversationContext for multi-turn conversation management.
grammar/            LSFB grammar modules: lsfb_rules, lsfb_crf, lsfb_translator.
trainer             Training loop with few-shot and full-retrain modes.
augmentation        Data augmentation strategies (train-only, no validation leakage).
dataset             Dataset loading and stratified split.
"""

from __future__ import annotations
