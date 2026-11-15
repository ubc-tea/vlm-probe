"""All-layer attribution patching from one backward equals per-layer replay."""
import importlib.util
from importlib.metadata import version
from pathlib import Path
import sys
import unittest

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "examples"))
from tiny_model import TinyModel
from vlm_probing import Prober, TokenMargin
from vlm_probing.api.common import evaluate
from vlm_probing.causal import AttributionPatching

HAS_V5 = (importlib.util.find_spec("transformers") is not None
          and version("transformers").split(".")[:2] == ["5", "3"])


def per_layer_reference(probe, inputs, source, metric, layers, tokens):
    """The previous algorithm: one gradient replay from each residual site."""
    sites = [probe.spec.residuals[i] for i in layers]
    trace = probe._run(inputs, capture=sites)
    layout = probe._layout(inputs, trace.logits)
    source_trace = probe._run(source, capture=sites)
    mask = layout.select(tokens)[..., None]
    scores, effects = [], []
    for site in sites:
        def replay(value, site=site):
            output = probe._run(inputs, interventions={site: lambda _: value}, grad=True)
            return evaluate(metric, output.logits, layout).double().sum()
        result = AttributionPatching().run(trace.activations[site], source_trace.activations[site],
                                           mask=mask, metric=replay)
        scores.append(result.tensors["attribution"].sum(-1))
        effects.append(result.tensors["estimated_effect"])
    return torch.stack(scores), torch.stack(effects)


class CountingProber:
    """Count gradient-enabled forwards issued through Prober._run."""

    def __init__(self, probe):
        self.calls, self._run = 0, probe._run
        probe._run = self

    def __call__(self, inputs, **kwargs):
        self.calls += bool(kwargs.get("grad"))
        return self._run(inputs, **kwargs)


class SingleBackwardAttributionTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(23)
        torch.set_num_threads(1)

    def check(self, model, inputs, source, metric, tokens):
        model.requires_grad_(False)
        probe = Prober(model)
        layers = sorted(probe.spec.residuals)
        counter = CountingProber(probe)
        result = probe.causal.attribute(layers=layers, tokens=tokens).run(
            inputs, source=source, metric=metric)
        self.assertEqual(counter.calls, 1)
        probe._run = counter._run
        scores, effects = per_layer_reference(probe, inputs, source, metric, layers, tokens)
        torch.testing.assert_close(result.tensors["token_scores"], scores.to(result.tensors["token_scores"]),
                                   atol=1e-6, rtol=1e-5)
        torch.testing.assert_close(result.tensors["estimated_effect"],
                                   effects.to(result.tensors["estimated_effect"]), atol=1e-6, rtol=1e-5)
        self.assertTrue((result.tensors["estimated_effect"].abs() > 0).any())
        self.assertTrue(all(param.grad is None for param in model.parameters()))
        # A layer subset keeps the same per-layer values.
        subset = probe.causal.attribute(layers=layers[1:], tokens=tokens).run(
            inputs, source=source, metric=metric)
        torch.testing.assert_close(subset.tensors["estimated_effect"], result.tensors["estimated_effect"][1:])

    def test_tiny_model_matches_per_layer_replay(self):
        inputs = {"input_ids": torch.tensor([[1, 2, 3], [4, 5, 6]]), "image_tokens": torch.randn(2, 2, 6)}
        source = {**inputs, "image_tokens": torch.randn(2, 2, 6)}
        self.check(TinyModel(), inputs, source, TokenMargin(4, 5), "visual")

    def test_metric_checks_match_per_layer_kernel(self):
        model = TinyModel().requires_grad_(False)
        probe = Prober(model)
        inputs = {"input_ids": torch.tensor([[1, 2, 3]]), "image_tokens": torch.randn(1, 2, 6)}
        source = {**inputs, "image_tokens": torch.randn(1, 2, 6)}
        method = probe.causal.attribute(layers=sorted(probe.spec.residuals))
        with self.assertRaisesRegex(ValueError, "finite"):
            method.run(inputs, source=source, metric=lambda logits: logits.sum() * float("inf"))
        with self.assertRaisesRegex(ValueError, "gradient must contain finite"):
            AttributionPatching().run(torch.ones(2), torch.zeros(2), gradient=torch.tensor([1.0, float("nan")]))
        with self.assertRaisesRegex(ValueError, "differentiable"):
            method.run(inputs, source=source, metric=lambda logits: torch.tensor(1.0))

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
        source = {**inputs, "pixel_values": -inputs["pixel_values"]}
        for tokens in ("visual", "all"):
            with self.subTest(tokens=tokens):
                self.check(model, inputs, source, TokenMargin(4, 5), tokens)


if __name__ == "__main__":
    unittest.main()
