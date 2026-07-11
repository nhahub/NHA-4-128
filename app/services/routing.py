import os
from typing import Optional, Dict, Any


class ModelRouter:
    def __init__(self):
        self._default_classifier = os.getenv(
            "CLASSIFIER_MODEL_URI", "hf_savedmodel"
        )
        self._default_segmenter = os.getenv(
            "SEGMENTER_MODEL_URI", "hf_savedmodel"
        )
        self._ab_enabled = os.getenv("AB_TESTING_ENABLED", "false").lower() == "true"
        self._ab_classifier_b = os.getenv("AB_CLASSIFIER_B")

    def resolve_classifier_version(self, requested: Optional[str] = None) -> str:
        if requested:
            return requested
        if self._ab_enabled and self._ab_classifier_b:
            import random
            return random.choice([self._default_classifier, self._ab_classifier_b])
        return self._default_classifier

    def get_ab_status(self) -> Dict[str, Any]:
        return {
            "enabled": self._ab_enabled,
            "classifier_a": self._default_classifier,
            "classifier_b": self._ab_classifier_b,
            "segmenter": self._default_segmenter,
        }


router = ModelRouter()
