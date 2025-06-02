# SPDX-License-Identifier: Apache-2.0
import numpy as np
import pytest
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.trainers import BpeTrainer
from tokenizers_python import PyTokenizer

try:
    from native_tokenizer_ext import encode
except ImportError:
    pytest.skip("native_tokenizer_ext not built", allow_module_level=True)


def create_test_tokenizer():
    """Create a simple BPE tokenizer for testing."""
    tokenizer = Tokenizer(BPE())
    tokenizer.pre_tokenizer = Whitespace()

    # train on some basic text
    trainer = BpeTrainer(vocab_size=1000, min_frequency=1)
    tokenizer.train_from_iterator(
        ["hello world", "test text", "encode function"], trainer)

    return tokenizer


def test_encode_basic():
    """Test basic encoding functionality."""
    tokenizer = create_test_tokenizer()
    py_tokenizer = PyTokenizer(tokenizer)

    text = "hello world"
    result = encode(py_tokenizer, text)

    # should return numpy array
    assert isinstance(result, np.ndarray)
    assert result.dtype == np.int64
    assert len(result.shape) == 1  # 1D array
    assert result.size > 0  # should have some tokens


def test_encode_empty_string():
    """Test encoding empty string."""
    tokenizer = create_test_tokenizer()
    py_tokenizer = PyTokenizer(tokenizer)

    result = encode(py_tokenizer, "")

    assert isinstance(result, np.ndarray)
    assert result.dtype == np.int64


def test_encode_consistency():
    """Test that same input produces same output."""
    tokenizer = create_test_tokenizer()
    py_tokenizer = PyTokenizer(tokenizer)

    text = "test consistency"
    result1 = encode(py_tokenizer, text)
    result2 = encode(py_tokenizer, text)

    np.testing.assert_array_equal(result1, result2)


def test_encode_vs_python_tokenizer():
    """Test that rust encode matches python tokenizer output."""
    tokenizer = create_test_tokenizer()
    py_tokenizer = PyTokenizer(tokenizer)

    text = "hello test"
    rust_result = encode(py_tokenizer, text)
    python_result = tokenizer.encode(text, add_special_tokens=False)

    expected = np.array(python_result.ids, dtype=np.int64)
    np.testing.assert_array_equal(rust_result, expected)


if __name__ == "__main__":
    pytest.main([__file__])
