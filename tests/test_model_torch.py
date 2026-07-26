import pytest

torch = pytest.importorskip("torch")

from lmlab.model import TinyGPT
from lmlab.multimodal import reliability_fusion


def test_model_shapes_tied_weights_and_gradients():
    torch.manual_seed(7)
    model = TinyGPT(64, 16, width=32, layers=2, heads=4)
    token_ids = torch.randint(0, 64, (2, 8))
    logits, loss = model(token_ids, token_ids)
    assert logits.shape == (2, 8, 64)
    assert model.head.weight is model.token_embedding.weight
    loss.backward()
    assert model.token_embedding.weight.grad is not None
    assert torch.isfinite(model.token_embedding.weight.grad).all()


def test_causal_prefix_invariance():
    torch.manual_seed(7)
    model = TinyGPT(64, 16, width=32, layers=2, heads=4).eval()
    first = torch.tensor([[1, 2, 3, 4, 5]])
    second = torch.tensor([[1, 2, 3, 30, 31]])
    logits_first, _ = model(first)
    logits_second, _ = model(second)
    assert torch.allclose(logits_first[:, :3], logits_second[:, :3], atol=1e-6)


def test_checkpoint_reload(tmp_path):
    torch.manual_seed(7)
    model = TinyGPT(32, 8, width=16, layers=1, heads=4).eval()
    ids = torch.randint(0, 32, (1, 6))
    expected, _ = model(ids)
    path = tmp_path / "model.pt"
    torch.save(model.state_dict(), path)
    restored = TinyGPT(32, 8, width=16, layers=1, heads=4).eval()
    restored.load_state_dict(torch.load(path, weights_only=True))
    actual, _ = restored(ids)
    assert torch.allclose(expected, actual)


def test_reliability_fusion():
    embeddings = torch.tensor([[[1.0, 0.0], [0.0, 1.0]]])
    reliability = torch.tensor([[3.0, 1.0]])
    fused, weights = reliability_fusion(embeddings, reliability)
    assert fused.shape == (1, 2)
    assert torch.allclose(weights.sum(dim=1), torch.ones(1))

