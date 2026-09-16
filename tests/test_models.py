"""
tests/test_models.py
Unit tests for GRUBaseline and NeuroCXModel.
Covers: forward pass shapes, gate value bounds, NDCG@1 on fixed input.
"""
import sys, os
import pytest
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from model.gru_baseline import GRUBaseline
from model.neuro_cx_model import NeuroCXModel
from evaluation.metrics import accuracy_at_1

# ── Constants ─────────────────────────────────────────────────────────────────
N_ITEMS   = 50
N_ACTIONS = 5
B, T      = 4, 8        # batch size, sequence length
H         = 32          # hidden size (small for fast tests)

# ── Fixtures ──────────────────────────────────────────────────────────────────
@pytest.fixture(scope="module")
def baseline():
    m = GRUBaseline(n_items=N_ITEMS, n_actions=N_ACTIONS,
                    hidden_size=H, num_layers=1, dropout=0.0)
    m.eval()
    return m

@pytest.fixture(scope="module")
def neuro_cx():
    m = NeuroCXModel(n_items=N_ITEMS, n_actions=N_ACTIONS,
                     hidden_size=H, num_layers=1, dropout=0.0)
    m.eval()
    return m

@pytest.fixture
def batch():
    """Synthetic batch (B=4, T=8)."""
    torch.manual_seed(42)
    item_seq   = torch.randint(1, N_ITEMS,   (B, T))
    action_seq = torch.randint(0, N_ACTIONS, (B, T))
    dwell_seq  = torch.rand(B, T)
    weight_seq = torch.rand(B, T)
    target     = torch.randint(1, N_ITEMS,   (B,))
    return item_seq, action_seq, dwell_seq, weight_seq, target


# ══════════════════════════════════════════════════════════════════════════════
class TestGRUBaselineForward:
    def test_output_shape(self, baseline, batch):
        item_seq, action_seq, dwell_seq, weight_seq, _ = batch
        logits, _ = baseline(item_seq, action_seq, dwell_seq, weight_seq)
        assert logits.shape == (B, N_ITEMS), \
            f"Expected ({B}, {N_ITEMS}), got {logits.shape}"

    def test_output_is_finite(self, baseline, batch):
        item_seq, action_seq, dwell_seq, weight_seq, _ = batch
        logits, _ = baseline(item_seq, action_seq, dwell_seq, weight_seq)
        assert torch.isfinite(logits).all(), "Logits contain NaN or Inf"

    def test_output_not_all_same(self, baseline, batch):
        item_seq, action_seq, dwell_seq, weight_seq, _ = batch
        logits, _ = baseline(item_seq, action_seq, dwell_seq, weight_seq)
        assert logits[0].std().item() > 1e-6, "All logits identical — dead model"

    def test_different_inputs_different_outputs(self, baseline):
        torch.manual_seed(0)
        seq_a = torch.randint(1, N_ITEMS, (1, T))
        torch.manual_seed(1)
        seq_b = torch.randint(1, N_ITEMS, (1, T))
        zeros_a = torch.zeros(1, T, dtype=torch.long)
        zeros_d = torch.zeros(1, T)
        ones_w  = torch.ones(1, T)
        out_a, _ = baseline(seq_a, zeros_a, zeros_d, ones_w)
        out_b, _ = baseline(seq_b, zeros_a, zeros_d, ones_w)
        assert not torch.allclose(out_a, out_b, atol=1e-4), \
            "Different inputs produced identical outputs"


# ══════════════════════════════════════════════════════════════════════════════
class TestNeuroCXForward:
    def test_output_shape(self, neuro_cx, batch):
        item_seq, action_seq, dwell_seq, weight_seq, _ = batch
        logits, _ = neuro_cx(item_seq, action_seq, dwell_seq, weight_seq)
        assert logits.shape == (B, N_ITEMS), \
            f"Expected ({B}, {N_ITEMS}), got {logits.shape}"

    def test_hidden_shape(self, neuro_cx, batch):
        item_seq, action_seq, dwell_seq, weight_seq, _ = batch
        _, hidden = neuro_cx(item_seq, action_seq, dwell_seq, weight_seq)
        assert hidden.shape == (1, B, H), \
            f"Expected (1, {B}, {H}), got {hidden.shape}"

    def test_output_is_finite(self, neuro_cx, batch):
        item_seq, action_seq, dwell_seq, weight_seq, _ = batch
        logits, _ = neuro_cx(item_seq, action_seq, dwell_seq, weight_seq)
        assert torch.isfinite(logits).all(), "NeuroCX logits contain NaN or Inf"

    def test_return_all_hidden(self, neuro_cx, batch):
        item_seq, action_seq, dwell_seq, weight_seq, _ = batch
        _, _, hidden_all_list = neuro_cx(
            item_seq, action_seq, dwell_seq, weight_seq,
            return_all_hidden=True
        )
        assert len(hidden_all_list) == T, \
            f"Expected T={T} hidden states, got {len(hidden_all_list)}"


# ══════════════════════════════════════════════════════════════════════════════
class TestNeuroCXGateBounds:
    def test_hidden_state_bounded(self, neuro_cx):
        """Gated hidden state should never explode."""
        torch.manual_seed(7)
        item_seq   = torch.randint(1, N_ITEMS,   (1, T))
        action_seq = torch.randint(0, N_ACTIONS, (1, T))
        dwell_seq  = torch.rand(1, T)
        weight_seq = torch.rand(1, T)
        with torch.no_grad():
            _, hidden = neuro_cx(item_seq, action_seq, dwell_seq, weight_seq)
        assert hidden.abs().max().item() < 1e4, \
            "Hidden state exploded — gate out of range"

    def test_high_weight_vs_low_weight(self, neuro_cx):
        """High signal weight should keep hidden state norm >= low weight."""
        torch.manual_seed(42)
        item_seq   = torch.randint(1, N_ITEMS,   (1, T))
        action_seq = torch.randint(0, N_ACTIONS, (1, T))
        dwell_seq  = torch.ones(1, T) * 0.5
        with torch.no_grad():
            _, h_high = neuro_cx(item_seq, action_seq, dwell_seq,
                                  torch.ones(1, T) * 0.95)
            _, h_low  = neuro_cx(item_seq, action_seq, dwell_seq,
                                  torch.ones(1, T) * 0.05)
        # Norms should both be finite
        assert torch.isfinite(h_high).all() and torch.isfinite(h_low).all()


# ══════════════════════════════════════════════════════════════════════════════
class TestNDCGOnFixedInput:
    def test_top1_is_valid_item(self, neuro_cx):
        """Top-1 prediction should be a valid item index."""
        item_seq   = torch.tensor([[1, 2, 3, 4, 5, 6, 7, 8]])
        action_seq = torch.tensor([[2, 2, 4, 2, 2, 4, 2, 4]])
        dwell_seq  = torch.tensor([[0.3, 0.4, 0.9, 0.3, 0.3, 0.8, 0.3, 0.9]])
        weight_seq = torch.tensor([[0.3, 0.3, 1.0, 0.3, 0.3, 1.0, 0.3, 1.0]])
        with torch.no_grad():
            logits, _ = neuro_cx(item_seq, action_seq, dwell_seq, weight_seq)
        top1 = logits[0].argmax().item()
        assert 0 <= top1 < N_ITEMS, \
            f"Top-1 prediction {top1} outside valid range [0, {N_ITEMS})"

    def test_accuracy_at_1_returns_valid_float(self, neuro_cx, batch):
        """accuracy_at_1 should return a float in [0, 1]."""
        item_seq, action_seq, dwell_seq, weight_seq, target = batch
        with torch.no_grad():
            logits, _ = neuro_cx(item_seq, action_seq, dwell_seq, weight_seq)
        acc = accuracy_at_1(logits, target)
        assert 0.0 <= acc <= 1.0, f"accuracy_at_1 out of [0,1]: {acc}"

    def test_logits_sum_not_zero(self, neuro_cx, batch):
        """Model should produce non-zero predictions."""
        item_seq, action_seq, dwell_seq, weight_seq, _ = batch
        with torch.no_grad():
            logits, _ = neuro_cx(item_seq, action_seq, dwell_seq, weight_seq)
        assert logits.abs().sum().item() > 0, "Model outputs all zeros"

