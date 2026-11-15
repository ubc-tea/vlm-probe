"""Path-integrated attribution patching: endpoints are measured, the estimate converges."""
import importlib.util
from importlib.metadata import version
from pathlib import Path
import sys
import unittest

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "examples"))
from tiny_model import TinyModel
from vlm_probing import Prober, TokenMargin
from vlm_probing.causal import AttributionPatching

HAS_V5 = (importlib.util.find_spec("transformers") is not None
          and version("transformers").split(".")[:2] == ["5", "3"])


class PathKernelTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(5)

    def test_quadratic_metric_is_exact_with_one_interval(self):
        # The gradient of a quadratic is linear along the path; the trapezoid is exact.
        a, b = torch.randn(3, 4, dtype=torch.float64), torch.randn(3, 4, dtype=torch.float64)
        metric = lambda x: (a * x * x + b * x).sum()
        receiver, source = torch.randn(3, 4, dtype=torch.float64), torch.randn(3, 4, dtype=torch.float64)
        result = AttributionPatching().run(receiver, source, metric=metric, steps=1)
        effect = metric(source) - metric(receiver)
        torch.testing.assert_close(result.tensors["effect"], effect)
        torch.testing.assert_close(result.tensors["estimated_effect"], effect)
        self.assertEqual(result.metadata["approximation"], "path_integrated")

    def test_error_shrinks_with_steps_and_first_order_is_unchanged(self):
        metric = lambda x: torch.tanh(3 * x).sum()
        receiver, source = torch.randn(2, 5, dtype=torch.float64), torch.randn(2, 5, dtype=torch.float64)
        errors = [AttributionPatching().run(receiver, source, metric=metric, steps=k)
                  .tensors["completeness_error"].abs().item() for k in (1, 4, 16, 64)]
        self.assertTrue(all(x > y for x, y in zip(errors, errors[1:])), errors)
        self.assertLess(errors[-1], 1e-3)
        first = AttributionPatching().run(receiver, source, metric=metric)
        point = receiver.clone().requires_grad_(True)
        grad = torch.autograd.grad(metric(point), point)[0]
        torch.testing.assert_close(first.tensors["estimated_effect"], ((source - receiver) * grad).sum())
        self.assertNotIn("effect", first.tensors)

    def test_mask_selects_the_replaced_entries(self):
        metric = lambda x: (x.sin() * x).sum()
        receiver, source = torch.randn(2, 3, 4), torch.randn(2, 3, 4)
        mask = torch.tensor([[True, False, True], [False, False, True]])[..., None]
        result = AttributionPatching().run(receiver, source, metric=metric, mask=mask, steps=8)
        expected = metric(torch.where(mask, source, receiver)) - metric(receiver)
        torch.testing.assert_close(result.tensors["effect"], expected)
        self.assertTrue((result.tensors["attribution"][~mask.expand_as(receiver)] == 0).all())

    def test_bfloat16_endpoint_is_the_exact_replacement(self):
        seen = []
        metric = lambda x: (seen.append(x.detach().clone()) or (x.float() ** 3).sum())
        receiver = torch.randn(4, 8).bfloat16()
        source = (receiver + 1e-3 * torch.randn(4, 8)).bfloat16()
        mask = torch.tensor([True, False, True, False])[:, None]
        for selection, expected in ((None, source), (mask, torch.where(mask, source, receiver))):
            seen.clear()
            AttributionPatching().run(receiver, source, metric=metric, mask=selection, steps=4)
            self.assertTrue(all(value.dtype == torch.bfloat16 for value in seen))
            self.assertTrue(torch.equal(seen[0], receiver) and torch.equal(seen[-1], expected))
            # Unselected rows stay bit-identical to the receiver at every path point.
            if selection is not None:
                self.assertTrue(all(torch.equal(value[~mask[:, 0]], receiver[~mask[:, 0]]) for value in seen))

    def test_invalid_arguments(self):
        receiver, source = torch.randn(2, 3), torch.randn(2, 3)
        metric = lambda x: x.pow(2).sum()
        for steps in (0, -1, 1.5, True):
            with self.assertRaisesRegex(ValueError, "steps"):
                AttributionPatching().run(receiver, source, metric=metric, steps=steps)
        with self.assertRaisesRegex(ValueError, "replay metric"):
            AttributionPatching().run(receiver, source, gradient=torch.ones(2, 3), steps=2)


class PathApiTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(11)
        torch.set_num_threads(1)

    def check(self, model, inputs, source, metric, tokens):
        model.requires_grad_(False)
        probe = Prober(model)
        layers = sorted(probe.spec.residuals)
        exact = probe.causal.patch(layers=layers, tokens=tokens).run(inputs, source=source, metric=metric)
        measured = exact.tensors["effect"].sum(-1)
        errors = []
        for steps in (1, 4, 32):
            result = probe.causal.attribute(layers=layers, tokens=tokens, steps=steps).run(
                inputs, source=source, metric=metric)
            # The path's endpoints are real replays: the same single-site replacement as patch.
            torch.testing.assert_close(result.tensors["effect"], measured.to(result.tensors["effect"]),
                                       atol=1e-5, rtol=1e-4)
            torch.testing.assert_close(result.tensors["completeness_error"],
                                       result.tensors["estimated_effect"] - result.tensors["effect"])
            torch.testing.assert_close(result.tensors["token_scores"].sum((-2, -1)),
                                       result.tensors["estimated_effect"], atol=1e-5, rtol=1e-4)
            self.assertEqual(result.metadata["configuration"]["steps"], steps)
            errors.append(result.tensors["completeness_error"].abs().max().item())
        # The first-order-like single interval must be visibly wrong here, and the path must fix it.
        self.assertGreater(errors[0], 1e-2 * measured.abs().max().item())
        self.assertLess(errors[-1], 0.05 * errors[0])
        self.assertTrue(all(param.grad is None for param in model.parameters()))
        first = probe.causal.attribute(layers=layers, tokens=tokens).run(inputs, source=source, metric=metric)
        self.assertEqual(set(first.tensors), {"token_scores", "estimated_effect"})

    def test_tiny_model(self):
        inputs = {"input_ids": torch.tensor([[1, 2, 3], [4, 5, 6]]), "image_tokens": torch.randn(2, 2, 6)}
        source = {**inputs, "image_tokens": 3 * torch.randn(2, 2, 6)}
        self.check(TinyModel(), inputs, source, TokenMargin(4, 5), "visual")

    def test_invalid_steps(self):
        probe = Prober(TinyModel())
        for steps in (0, 2.0, False):
            with self.assertRaisesRegex(ValueError, "steps"):
                probe.causal.attribute(steps=steps)

    @unittest.skipUnless(HAS_V5, "native Qwen3-VL requires Transformers 5.3.x")
    def test_qwen3_vl_with_several_deepstack_levels(self):
        import transformers as hf

        text = dict(vocab_size=40, hidden_size=16, intermediate_size=24, num_hidden_layers=4,
                    num_attention_heads=2, num_key_value_heads=1, pad_token_id=0, head_dim=8,
                    rope_parameters=dict(rope_type="default", mrope_section=[1, 1, 2]))
        vision = dict(depth=3, hidden_size=16, intermediate_size=24, num_heads=2, patch_size=2,
                      temporal_patch_size=1, spatial_merge_size=2, out_hidden_size=16,
                      num_position_embeddings=16, deepstack_visual_indexes=[0, 1, 2])
        config = hf.Qwen3VLConfig(text_config=text, vision_config=vision, image_token_id=28,
                                  video_token_id=30, vision_start_token_id=27, vision_end_token_id=29)
        config._attn_implementation = "eager"
        model = hf.Qwen3VLForConditionalGeneration(config).eval()
        inputs = {"input_ids": torch.tensor([[1, 27, 28, 29, 3, 0], [1, 27, 28, 28, 29, 3]]),
                  "attention_mask": torch.tensor([[1, 1, 1, 1, 1, 0], [1, 1, 1, 1, 1, 1]]),
                  "image_grid_thw": torch.tensor([[1, 2, 2], [1, 2, 4]]),
                  "pixel_values": torch.randn(12, 12)}
        source = {**inputs, "pixel_values": -3 * inputs["pixel_values"]}
        for tokens in ("visual", "all"):
            with self.subTest(tokens=tokens):
                self.check(model, inputs, source, TokenMargin(4, 5), tokens)


if __name__ == "__main__":
    unittest.main()
