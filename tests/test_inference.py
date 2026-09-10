"""
Unit and Integration Tests for FaceSense AI Inference Pipeline
Tests checkpoint loading, model reconstruction, image preprocessing, output shapes,
probability sum, valid emotion labels, confidence score range, error handling, and thresholding.
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import unittest
import numpy as np
from PIL import Image
import torch

from ml.inference.predictor import (
    EmotionPredictor,
    get_inference_transforms,
    load_and_preprocess_image,
)


class TestInferencePipeline(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.checkpoint_path = PROJECT_ROOT / "ml" / "models" / "checkpoints" / "final" / "best_model.pt"
        cls.test_data_dir = PROJECT_ROOT / "Data(FER2013)" / "test"
        
        # Pick one real image from test set
        cls.real_img_path = next(cls.test_data_dir.rglob("*.jpg"))
        cls.predictor = EmotionPredictor(checkpoint_path=cls.checkpoint_path)

    def test_checkpoint_exists_and_loads(self):
        """Verify the final checkpoint exists and predictor instantiates without error."""
        self.assertTrue(self.checkpoint_path.exists())
        self.assertEqual(self.predictor.num_classes, 7)
        self.assertEqual(self.predictor.model_name, "ResidualEmotionCNN")
        self.assertFalse(self.predictor.model.training)

    def test_image_preprocessing_pipeline(self):
        """Verify image preprocessing produces [1, 1, 48, 48] float tensor and valid PIL image."""
        tensor, vis_pil = load_and_preprocess_image(self.real_img_path)
        self.assertEqual(tensor.shape, (1, 1, 48, 48))
        self.assertEqual(tensor.dtype, torch.float32)
        self.assertEqual(vis_pil.size, (48, 48))
        self.assertEqual(vis_pil.mode, "L")

        # Test normalization range: normalized with mean=0.5, std=0.5 -> [-1.0, 1.0]
        self.assertTrue(torch.all(tensor >= -1.05))
        self.assertTrue(torch.all(tensor <= 1.05))

    def test_prediction_output_structure_and_shape(self):
        """Verify prediction dictionary keys, data types, and valid emotion label."""
        result = self.predictor.predict(self.real_img_path)

        self.assertIn("predicted_emotion", result)
        self.assertIn("confidence", result)
        self.assertIn("probabilities", result)
        self.assertIn("raw_predicted_emotion", result)
        self.assertIn("is_uncertain", result)
        self.assertIn("model_name", result)
        self.assertIn("checkpoint_path", result)

        self.assertIn(result["predicted_emotion"], self.predictor.class_names)
        self.assertIn(result["raw_predicted_emotion"], self.predictor.class_names)
        self.assertIsInstance(result["confidence"], float)
        self.assertGreaterEqual(result["confidence"], 0.0)
        self.assertLessEqual(result["confidence"], 1.0)

    def test_probabilities_sum_to_one(self):
        """Verify softmax probabilities sum to approximately 1.0 and cover all 7 classes."""
        result = self.predictor.predict(self.real_img_path)
        probs = result["probabilities"]
        
        self.assertEqual(len(probs), 7)
        for class_name in self.predictor.class_names:
            self.assertIn(class_name, probs)
            self.assertGreaterEqual(probs[class_name], 0.0)
            self.assertLessEqual(probs[class_name], 1.0)

        total_prob = sum(probs.values())
        self.assertAlmostEqual(total_prob, 1.0, places=4)

    def test_confidence_threshold_uncertainty(self):
        """Verify that high threshold forces 'uncertain' prediction."""
        # Threshold at 0.9999 should make predictions uncertain
        result_uncertain = self.predictor.predict(self.real_img_path, confidence_threshold=0.9999)
        self.assertEqual(result_uncertain["predicted_emotion"], "uncertain")
        self.assertTrue(result_uncertain["is_uncertain"])
        self.assertIn(result_uncertain["raw_predicted_emotion"], self.predictor.class_names)

        # Threshold at 0.0 should not be uncertain
        result_certain = self.predictor.predict(self.real_img_path, confidence_threshold=0.0)
        self.assertNotEqual(result_certain["predicted_emotion"], "uncertain")
        self.assertFalse(result_certain["is_uncertain"])

    def test_input_types_support(self):
        """Verify prediction supports str, Path, PIL Image, and numpy arrays."""
        # PIL Image
        pil_img = Image.open(self.real_img_path)
        res_pil = self.predictor.predict(pil_img)
        self.assertIn(res_pil["predicted_emotion"], self.predictor.class_names)

        # Numpy Array RGB
        np_arr = np.array(pil_img.convert("RGB"))
        res_np = self.predictor.predict(np_arr)
        self.assertIn(res_np["predicted_emotion"], self.predictor.class_names)

        # Predict with visual
        res_vis, vis_face = self.predictor.predict_with_visual(self.real_img_path)
        self.assertEqual(vis_face.size, (48, 48))
        self.assertIn(res_vis["predicted_emotion"], self.predictor.class_names)

    def test_error_handling_missing_and_invalid_files(self):
        """Verify informative errors on missing or invalid image files."""
        # Missing file
        with self.assertRaises(FileNotFoundError):
            self.predictor.predict("non_existent_image_12345.jpg")

        # Unsupported input type
        with self.assertRaises(TypeError):
            self.predictor.predict(12345)

        # Missing checkpoint
        with self.assertRaises(FileNotFoundError):
            EmotionPredictor(checkpoint_path="non_existent_checkpoint.pt")


if __name__ == "__main__":
    unittest.main()
