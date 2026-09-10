"""
FaceSense AI - User Feedback and Continuous Improvement Module
Enables structured collection, validation, and storage of facial expression feedback records
for human-in-the-loop review and continuous dataset building.
"""

from ml.feedback.collector import FeedbackCollector, FeedbackState
from ml.feedback.dataset_builder import FeedbackDatasetBuilder

__all__ = [
    "FeedbackCollector",
    "FeedbackState",
    "FeedbackDatasetBuilder",
]
