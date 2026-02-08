import torch
from foundations.micrograd.engine import Value


def test_sanity_check():
    """Test a simple expression against PyTorch."""
    x = Value(-4.0)
    z = 2 * x + 2 + x
    q = z.relu() + z * x
    h = (z * z).relu()
    y = h + q + q * x
    y.backward()
    xmg, ymg = x, y

    x = torch.tensor([-4.0], dtype=torch.float64, requires_grad=True)
    z = 2 * x + 2 + x
    q = z.relu() + z * x
    h = (z * z).relu()
    y = h + q + q * x
    y.backward()
    xpt, ypt = x, y

    # Forward pass values should match
    assert abs(ymg.data - ypt.item()) < 1e-6
    # Backward pass gradients should match
    assert abs(xmg.grad - xpt.grad.item()) < 1e-6


def test_more_ops():
    """Test a wider range of operations against PyTorch."""
    a = Value(-4.0)
    b = Value(2.0)
    c = a + b
    d = a * b + b**3
    c += c + 1
    c += 1 + c + (-a)
    d += d * 2 + (b + a).relu()
    d += 3 * d + (b - a).relu()
    e = c - d
    f = e**2
    g = f / 2.0
    g += 10.0 / f
    g.backward()
    amg, bmg, gmg = a, b, g

    a = torch.tensor([-4.0], dtype=torch.float64, requires_grad=True)
    b = torch.tensor([2.0], dtype=torch.float64, requires_grad=True)
    c = a + b
    d = a * b + b**3
    c = c + c + 1
    c = c + 1 + c + (-a)
    d = d + d * 2 + (b + a).relu()
    d = d + 3 * d + (b - a).relu()
    e = c - d
    f = e**2
    g = f / 2.0
    g = g + 10.0 / f
    g.backward()
    apt, bpt, gpt = a, b, g

    tol = 1e-6
    # Forward pass
    assert abs(gmg.data - gpt.item()) < tol
    # Backward pass
    assert abs(amg.grad - apt.grad.item()) < tol
    assert abs(bmg.grad - bpt.grad.item()) < tol


def test_tanh():
    """Test tanh against PyTorch."""
    x = Value(0.8)
    y = x.tanh()
    y.backward()

    xpt = torch.tensor([0.8], dtype=torch.float64, requires_grad=True)
    ypt = xpt.tanh()
    ypt.backward()

    assert abs(y.data - ypt.item()) < 1e-6
    assert abs(x.grad - xpt.grad.item()) < 1e-6
