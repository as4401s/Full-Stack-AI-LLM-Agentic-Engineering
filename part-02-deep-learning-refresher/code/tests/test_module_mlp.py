"""Tests for the nn.Module network: parameter count, dropout modes, schedule and a short training run."""
import torch

from dl_refresher.data import make_datasets
from dl_refresher.module_mlp import MLP, count_parameters, train_module, warmup_cosine


def test_parameter_count_is_82():
    assert count_parameters(MLP((16,))) == 82         # 2*16 + 16 + 16*2 + 2


def test_dropout_changes_outputs_only_in_train_mode():
    torch.manual_seed(0)
    model = MLP((32,), dropout=0.5)
    x = torch.randn(6, 2)
    model.train()
    assert not torch.equal(model(x), model(x))        # random units switched off each time
    model.eval()
    assert torch.equal(model(x), model(x))            # dropout is off, outputs repeat


def test_warmup_cosine_values():
    factor = warmup_cosine(warmup=2, total=10)
    assert abs(factor(0) - 0.5) < 1e-9 and abs(factor(1) - 1.0) < 1e-9
    assert abs(factor(6) - 0.5) < 1e-9                 # halfway down the cosine
    assert abs(factor(10)) < 1e-9


def test_adamw_training_lowers_loss():
    train, val = make_datasets()
    torch.manual_seed(0)
    out = train_module(train, val, MLP((16,)), optimizer="adamw", lr=0.01, steps=100)
    assert out["history"]["train_loss"][-1] < 0.6 * out["history"]["train_loss"][0]
