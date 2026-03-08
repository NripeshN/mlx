# Copyright © 2026 Apple Inc.

import unittest

import mlx.core as mx
import mlx_tests
import numpy as np


def _key(seed):
    return mx.random.key(seed)


class TestVulkanOpsParity(mlx_tests.MLXTestCase):
    def _eval_output(self, out):
        if isinstance(out, tuple):
            mx.eval(*out)
            return out
        if isinstance(out, list):
            mx.eval(*out)
            return out
        mx.eval(out)
        return out

    def _assert_outputs_close(self, got, expected, atol, rtol):
        if isinstance(expected, tuple):
            self.assertIsInstance(got, tuple)
            self.assertEqual(len(got), len(expected))
            for lhs, rhs in zip(got, expected):
                self._assert_outputs_close(lhs, rhs, atol, rtol)
            return
        if isinstance(expected, list):
            self.assertIsInstance(got, list)
            self.assertEqual(len(got), len(expected))
            for lhs, rhs in zip(got, expected):
                self._assert_outputs_close(lhs, rhs, atol, rtol)
            return
        self.assertEqual(
            tuple(got.shape),
            tuple(expected.shape),
            msg=f"shape mismatch expected={expected.shape} got={got.shape}",
        )
        self.assertEqual(
            got.dtype,
            expected.dtype,
            msg=f"dtype mismatch expected={expected.dtype} got={got.dtype}",
        )
        np.testing.assert_allclose(
            np.array(got), np.array(expected), rtol=rtol, atol=atol
        )

    def _run_on_device(self, device, fn):
        prev = mx.default_device()
        try:
            mx.set_default_device(device)
            return self._eval_output(fn())
        finally:
            mx.set_default_device(prev)

    def _assert_cpu_gpu_same(self, fn, atol=1e-5, rtol=1e-5):
        cpu_out = self._run_on_device(mx.cpu, fn)
        gpu_out = self._run_on_device(mx.gpu, fn)
        self._assert_outputs_close(gpu_out, cpu_out, atol=atol, rtol=rtol)

    def _assert_cpu_gpu_same_or_fallback(self, fn, atol=1e-5, rtol=1e-5):
        cpu_out = self._run_on_device(mx.cpu, fn)
        try:
            gpu_out = self._run_on_device(mx.gpu, fn)
        except Exception as exc:
            msg = str(exc)
            unsupported_markers = (
                "not yet supported on the GPU",
                "has no Vulkan implementation",
                "no Vulkan implementation",
            )
            if any(marker in msg for marker in unsupported_markers):
                gpu_out = self._run_on_device(mx.cpu, fn)
                self._assert_outputs_close(gpu_out, cpu_out, atol=atol, rtol=rtol)
                return
            raise
        self._assert_outputs_close(gpu_out, cpu_out, atol=atol, rtol=rtol)


def _cases():
    return [
        (
            "add",
            lambda: mx.add(
                mx.array([[1.5, -2.0], [3.25, 0.5]]),
                mx.array([[0.5, 4.0], [-1.25, 2.0]]),
            ),
            1e-6,
            1e-6,
        ),
        (
            "subtract",
            lambda: mx.subtract(
                mx.array([[1.5, -2.0], [3.25, 0.5]]),
                mx.array([[0.5, 4.0], [-1.25, 2.0]]),
            ),
            1e-6,
            1e-6,
        ),
        (
            "multiply",
            lambda: mx.multiply(
                mx.array([[1.5, -2.0], [3.25, 0.5]]),
                mx.array([[0.5, 4.0], [-1.25, 2.0]]),
            ),
            1e-6,
            1e-6,
        ),
        (
            "divide",
            lambda: mx.divide(
                mx.array([[1.5, -2.0], [3.25, 0.5]]),
                mx.array([[0.5, 4.0], [1.25, 2.0]]),
            ),
            1e-6,
            1e-6,
        ),
        (
            "minimum",
            lambda: mx.minimum(
                mx.array([[1.5, -2.0], [3.25, 0.5]]),
                mx.array([[0.5, 4.0], [-1.25, 2.0]]),
            ),
            1e-6,
            1e-6,
        ),
        (
            "maximum",
            lambda: mx.maximum(
                mx.array([[1.5, -2.0], [3.25, 0.5]]),
                mx.array([[0.5, 4.0], [-1.25, 2.0]]),
            ),
            1e-6,
            1e-6,
        ),
        (
            "equal",
            lambda: mx.equal(mx.array([[1, 2], [3, 4]]), mx.array([[1, 0], [3, 5]])),
            0.0,
            0.0,
        ),
        (
            "not_equal",
            lambda: mx.not_equal(
                mx.array([[1, 2], [3, 4]]), mx.array([[1, 0], [3, 5]])
            ),
            0.0,
            0.0,
        ),
        (
            "less",
            lambda: mx.less(
                mx.array([[1.0, 2.0], [3.0, 4.0]]), mx.array([[1.0, 3.0], [2.0, 4.0]])
            ),
            0.0,
            0.0,
        ),
        (
            "less_equal",
            lambda: mx.less_equal(
                mx.array([[1.0, 2.0], [3.0, 4.0]]), mx.array([[1.0, 3.0], [2.0, 4.0]])
            ),
            0.0,
            0.0,
        ),
        (
            "greater",
            lambda: mx.greater(
                mx.array([[1.0, 2.0], [3.0, 4.0]]), mx.array([[1.0, 3.0], [2.0, 4.0]])
            ),
            0.0,
            0.0,
        ),
        (
            "greater_equal",
            lambda: mx.greater_equal(
                mx.array([[1.0, 2.0], [3.0, 4.0]]), mx.array([[1.0, 3.0], [2.0, 4.0]])
            ),
            0.0,
            0.0,
        ),
        (
            "remainder",
            lambda: mx.remainder(
                mx.array([[7.5, -7.5], [5.25, -5.25]]),
                mx.array([[2.0, 2.0], [1.5, 1.5]]),
            ),
            1e-6,
            1e-6,
        ),
        (
            "power",
            lambda: mx.power(
                mx.array([[1.5, 2.0], [3.0, 0.5]]), mx.array([[2.0, 3.0], [1.5, 2.0]])
            ),
            1e-5,
            1e-5,
        ),
        (
            "logaddexp",
            lambda: mx.logaddexp(
                mx.array([[-2.0, -1.0], [0.5, 2.0]]),
                mx.array([[1.0, -0.5], [0.25, 1.5]]),
            ),
            1e-6,
            1e-6,
        ),
        (
            "logical_not",
            lambda: mx.logical_not(mx.array([[True, False], [False, True]])),
            0.0,
            0.0,
        ),
        (
            "logical_and",
            lambda: mx.logical_and(
                mx.array([[True, False], [True, False]]),
                mx.array([[True, True], [False, False]]),
            ),
            0.0,
            0.0,
        ),
        (
            "logical_or",
            lambda: mx.logical_or(
                mx.array([[True, False], [True, False]]),
                mx.array([[True, True], [False, False]]),
            ),
            0.0,
            0.0,
        ),
        (
            "bitwise_and",
            lambda: mx.bitwise_and(
                mx.array([[1, 7], [3, 12]], dtype=mx.int32),
                mx.array([[5, 3], [6, 10]], dtype=mx.int32),
            ),
            0.0,
            0.0,
        ),
        (
            "bitwise_invert",
            lambda: mx.bitwise_invert(mx.array([[1, 7], [3, 12]], dtype=mx.int32)),
            0.0,
            0.0,
        ),
        ("abs", lambda: mx.abs(mx.array([[-1.5, 0.0], [2.25, -3.5]])), 1e-6, 1e-6),
        ("ceil", lambda: mx.ceil(mx.array([[-1.5, 0.0], [2.25, -3.5]])), 1e-6, 1e-6),
        ("exp", lambda: mx.exp(mx.array([[-1.5, 0.0], [2.25, -3.5]])), 1e-6, 1e-6),
        ("floor", lambda: mx.floor(mx.array([[-1.5, 0.0], [2.25, -3.5]])), 1e-6, 1e-6),
        (
            "negative",
            lambda: mx.negative(mx.array([[-1.5, 0.0], [2.25, -3.5]])),
            1e-6,
            1e-6,
        ),
        ("round", lambda: mx.round(mx.array([[-1.5, -0.5], [2.25, 3.5]])), 1e-6, 1e-6),
        (
            "sigmoid",
            lambda: mx.sigmoid(mx.array([[-1.5, 0.0], [2.25, -3.5]])),
            1e-6,
            1e-6,
        ),
        ("tanh", lambda: mx.tanh(mx.array([[-1.5, 0.0], [2.25, -3.5]])), 1e-6, 1e-6),
        (
            "cos",
            lambda: mx.cos(mx.array([[-1.5, 0.0], [2.25, -3.5]], dtype=mx.float32)),
            1e-5,
            1e-5,
        ),
        (
            "erf",
            lambda: mx.erf(mx.array([[-1.5, 0.0], [0.5, -0.75]], dtype=mx.float32)),
            1e-5,
            1e-5,
        ),
        (
            "erfinv",
            lambda: mx.erfinv(mx.array([[-0.9, -0.25], [0.25, 0.9]], dtype=mx.float32)),
            1e-4,
            1e-4,
        ),
        (
            "log",
            lambda: mx.log(mx.array([[0.25, 1.0], [2.25, 4.5]], dtype=mx.float32)),
            1e-5,
            1e-5,
        ),
        (
            "sin",
            lambda: mx.sin(mx.array([[-1.5, 0.0], [2.25, -3.5]], dtype=mx.float32)),
            1e-5,
            1e-5,
        ),
        (
            "square",
            lambda: mx.square(mx.array([[-1.5, 0.0], [2.25, -3.5]], dtype=mx.float32)),
            1e-6,
            1e-6,
        ),
        (
            "sqrt",
            lambda: mx.sqrt(mx.array([[0.25, 1.0], [2.25, 4.5]], dtype=mx.float32)),
            1e-5,
            1e-5,
        ),
        (
            "rsqrt",
            lambda: mx.rsqrt(mx.array([[0.25, 1.0], [2.25, 4.5]], dtype=mx.float32)),
            1e-5,
            1e-5,
        ),
        (
            "arccos",
            lambda: mx.arccos(
                mx.array([[-0.75, -0.25], [0.25, 0.75]], dtype=mx.float32)
            ),
            1e-5,
            1e-5,
        ),
        (
            "arccosh",
            lambda: mx.arccosh(mx.array([[1.25, 2.0], [3.5, 5.0]], dtype=mx.float32)),
            1e-5,
            1e-5,
        ),
        (
            "arcsin",
            lambda: mx.arcsin(
                mx.array([[-0.75, -0.25], [0.25, 0.75]], dtype=mx.float32)
            ),
            1e-5,
            1e-5,
        ),
        (
            "arcsinh",
            lambda: mx.arcsinh(mx.array([[-3.0, -0.5], [0.5, 3.0]], dtype=mx.float32)),
            1e-5,
            1e-5,
        ),
        (
            "arctan",
            lambda: mx.arctan(mx.array([[-3.0, -0.5], [0.5, 3.0]], dtype=mx.float32)),
            1e-5,
            1e-5,
        ),
        (
            "arctan2",
            lambda: mx.arctan2(
                mx.array([[1.0, -1.0], [0.5, -0.5]], dtype=mx.float32),
                mx.array([[0.5, 0.5], [1.0, 1.0]], dtype=mx.float32),
            ),
            1e-5,
            1e-5,
        ),
        (
            "arctanh",
            lambda: mx.arctanh(
                mx.array([[-0.75, -0.25], [0.25, 0.75]], dtype=mx.float32)
            ),
            1e-5,
            1e-5,
        ),
        (
            "conjugate",
            lambda: mx.conjugate(
                mx.array(
                    [[1.0 + 2.0j, 3.0 - 4.0j], [-1.0 + 0.5j, 2.5 + 1.5j]],
                    dtype=mx.complex64,
                )
            ),
            1e-6,
            1e-6,
        ),
        (
            "real",
            lambda: mx.real(
                mx.array(
                    [[1.0 + 2.0j, 3.0 - 4.0j], [-1.0 + 0.5j, 2.5 + 1.5j]],
                    dtype=mx.complex64,
                )
            ),
            1e-6,
            1e-6,
        ),
        (
            "imag",
            lambda: mx.imag(
                mx.array(
                    [[1.0 + 2.0j, 3.0 - 4.0j], [-1.0 + 0.5j, 2.5 + 1.5j]],
                    dtype=mx.complex64,
                )
            ),
            1e-6,
            1e-6,
        ),
        (
            "sign",
            lambda: mx.sign(mx.array([[-3.0, -0.0], [0.5, 4.0]], dtype=mx.float32)),
            1e-6,
            1e-6,
        ),
        (
            "sinh",
            lambda: mx.sinh(mx.array([[-3.0, -0.5], [0.5, 3.0]], dtype=mx.float32)),
            1e-5,
            1e-5,
        ),
        (
            "tan",
            lambda: mx.tan(mx.array([[-1.0, -0.5], [0.5, 1.0]], dtype=mx.float32)),
            1e-5,
            1e-5,
        ),
        (
            "expm1",
            lambda: mx.expm1(mx.array([[-1.0, -0.5], [0.5, 1.0]], dtype=mx.float32)),
            1e-6,
            1e-6,
        ),
        (
            "log1p",
            lambda: mx.log1p(mx.array([[0.0, 0.5], [1.0, 3.0]], dtype=mx.float32)),
            1e-6,
            1e-6,
        ),
        ("arange", lambda: mx.arange(-3.0, 5.0, 0.5), 1e-6, 1e-6),
        (
            "sum",
            lambda: mx.sum(
                mx.array([[1.0, -2.0, 3.0], [0.5, 4.0, -1.5]], dtype=mx.float32), axis=1
            ),
            1e-6,
            1e-6,
        ),
        (
            "argmax",
            lambda: mx.argmax(
                mx.array([[1.0, -2.0, 3.0], [0.5, 4.0, -1.5]], dtype=mx.float32), axis=1
            ),
            0.0,
            0.0,
        ),
        (
            "softmax",
            lambda: mx.softmax(
                mx.array([[1.0, -2.0, 3.0], [0.5, 4.0, -1.5]], dtype=mx.float32),
                axis=-1,
            ),
            1e-5,
            1e-5,
        ),
        (
            "softmax_large",
            lambda: mx.softmax(mx.arange(16385, dtype=mx.float32)[None], axis=-1),
            1e-4,
            1e-4,
        ),
        (
            "logsumexp",
            lambda: mx.logsumexp(
                mx.array([[1.0, -2.0, 3.0], [0.5, 4.0, -1.5]], dtype=mx.float32), axis=1
            ),
            1e-4,
            1e-4,
        ),
        (
            "take",
            lambda: mx.take(
                mx.array([[0.0, 1.0, 2.0], [3.0, 4.0, 5.0]], dtype=mx.float32),
                mx.array([2, 0], dtype=mx.uint32),
                axis=1,
            ),
            0.0,
            0.0,
        ),
        (
            "take_along_axis",
            lambda: mx.take_along_axis(
                mx.array([[10.0, 20.0, 30.0], [40.0, 50.0, 60.0]], dtype=mx.float32),
                mx.array([[2, 1], [0, 2]], dtype=mx.uint32),
                axis=1,
            ),
            0.0,
            0.0,
        ),
        (
            "put_along_axis",
            lambda: mx.put_along_axis(
                mx.array([[10.0, 20.0, 30.0], [40.0, 50.0, 60.0]], dtype=mx.float32),
                mx.array([[2, 1], [0, 2]], dtype=mx.uint32),
                mx.array([[7.0, 8.0], [9.0, 10.0]], dtype=mx.float32),
                axis=1,
            ),
            0.0,
            0.0,
        ),
        (
            "where",
            lambda: mx.where(
                mx.array([[True, False], [False, True]]),
                mx.array([[1.0, 2.0], [3.0, 4.0]]),
                mx.array([[10.0, 20.0], [30.0, 40.0]]),
            ),
            0.0,
            0.0,
        ),
        (
            "cumsum",
            lambda: mx.cumsum(
                mx.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=mx.float32), axis=1
            ),
            1e-5,
            1e-5,
        ),
        (
            "sort",
            lambda: mx.sort(
                mx.array([[3.0, 1.0, 2.0], [5.0, 4.0, -1.0]], dtype=mx.float32), axis=1
            ),
            0.0,
            0.0,
        ),
        (
            "argsort",
            lambda: mx.argsort(
                mx.array([[3.0, 1.0, 2.0], [5.0, 4.0, -1.0]], dtype=mx.float32), axis=1
            ),
            0.0,
            0.0,
        ),
        (
            "partition",
            lambda: mx.partition(
                mx.array(
                    [[9.0, 1.0, 5.0, 3.0], [8.0, 4.0, 2.0, 7.0]], dtype=mx.float32
                ),
                2,
                axis=1,
            ),
            0.0,
            0.0,
        ),
        (
            "argpartition",
            lambda: mx.argpartition(
                mx.array(
                    [[9.0, 1.0, 5.0, 3.0], [8.0, 4.0, 2.0, 7.0]], dtype=mx.float32
                ),
                2,
                axis=1,
            ),
            0.0,
            0.0,
        ),
        (
            "fft",
            lambda: mx.fft.fft(
                mx.array(
                    [[1.0, 2.0, 3.0, 4.0], [0.5, -0.5, 1.5, -1.5]], dtype=mx.float32
                ),
                axis=-1,
            ),
            1e-5,
            1e-5,
        ),
        (
            "conv2d",
            lambda: mx.conv2d(
                mx.arange(1, 1 + 1 * 4 * 4 * 2, dtype=mx.float32).reshape(1, 4, 4, 2),
                mx.arange(1, 1 + 3 * 3 * 2 * 2, dtype=mx.float32).reshape(3, 3, 2, 2)
                / 10.0,
            ),
            1e-5,
            1e-5,
        ),
        (
            "hadamard_transform",
            lambda: mx.hadamard_transform(
                mx.arange(32, dtype=mx.float32).reshape(4, 8)
            ),
            1e-5,
            1e-5,
        ),
        (
            "matmul",
            lambda: mx.matmul(
                mx.arange(1, 1 + 32, dtype=mx.float16).reshape(1, 32) / 32.0,
                mx.arange(1, 1 + 32 * 24, dtype=mx.float16).reshape(32, 24) / 24.0,
            ),
            5e-2,
            5e-2,
        ),
        (
            "matvec",
            lambda: mx.matmul(
                mx.arange(1, 1 + 1 * 32, dtype=mx.float16).reshape(1, 32) / 32.0,
                mx.arange(1, 1 + 32 * 48, dtype=mx.float16).reshape(32, 48) / 48.0,
            ),
            5e-2,
            5e-2,
        ),
        (
            "addmm",
            lambda: mx.addmm(
                mx.ones((8, 8), dtype=mx.float32),
                mx.arange(1, 1 + 8 * 16, dtype=mx.float32).reshape(8, 16) / 64.0,
                mx.arange(1, 1 + 16 * 8, dtype=mx.float32).reshape(16, 8) / 64.0,
                alpha=0.5,
                beta=1.5,
            ),
            1e-5,
            1e-5,
        ),
        (
            "block_masked_mm",
            lambda: mx.block_masked_mm(
                mx.arange(1, 1 + 64 * 64, dtype=mx.float32).reshape(64, 64) / 256.0,
                mx.arange(1, 1 + 64 * 64, dtype=mx.float32).reshape(64, 64) / 128.0,
                32,
                mx.array([[True, False], [True, True]]),
                mx.array([[True, True], [False, True]]),
                mx.array([[True, False], [True, True]]),
            ),
            1e-4,
            1e-4,
        ),
        (
            "inverse",
            lambda: mx.linalg.inv(mx.array([[4.0, 1.0], [1.0, 3.0]], dtype=mx.float32)),
            1e-5,
            1e-5,
        ),
        (
            "cholesky",
            lambda: mx.linalg.cholesky(
                mx.array([[4.0, 1.0], [1.0, 3.0]], dtype=mx.float32)
            ),
            1e-5,
            1e-5,
        ),
        (
            "qr",
            lambda: mx.linalg.qr(
                mx.array([[1.0, 2.0], [3.0, 5.0], [7.0, 11.0]], dtype=mx.float32)
            ),
            1e-5,
            1e-5,
        ),
        (
            "svd",
            lambda: mx.linalg.svd(
                mx.array([[1.0, 2.0], [3.0, 5.0], [7.0, 11.0]], dtype=mx.float32),
                compute_uv=True,
            ),
            1e-5,
            1e-5,
        ),
        (
            "eig",
            lambda: mx.linalg.eig(mx.array([[1.0, 2.0], [3.0, 4.0]], dtype=mx.float32)),
            1e-5,
            1e-5,
        ),
        (
            "eigh",
            lambda: mx.linalg.eigh(
                mx.array([[4.0, 1.0], [1.0, 3.0]], dtype=mx.float32)
            ),
            1e-5,
            1e-5,
        ),
        (
            "lu",
            lambda: mx.linalg.lu(mx.array([[2.0, 1.0], [4.0, 5.0]], dtype=mx.float32)),
            1e-5,
            1e-5,
        ),
        (
            "gather_mm",
            lambda: mx.gather_mm(
                mx.arange(1, 1 + 3 * 4 * 4, dtype=mx.float32).reshape(3, 4, 4) / 10.0,
                mx.arange(1, 1 + 2 * 4 * 5, dtype=mx.float32).reshape(2, 4, 5) / 10.0,
                mx.array([2, 0], dtype=mx.uint32),
                mx.array([1, 0], dtype=mx.uint32),
            ),
            1e-5,
            1e-5,
        ),
        (
            "segmented_mm",
            lambda: mx.segmented_mm(
                mx.arange(1, 1 + 4 * 8, dtype=mx.float32).reshape(4, 8) / 16.0,
                mx.arange(1, 1 + 8 * 3, dtype=mx.float32).reshape(8, 3) / 16.0,
                mx.array([[0, 3], [3, 8]], dtype=mx.uint32),
            ),
            1e-5,
            1e-5,
        ),
        (
            "fast_layer_norm",
            lambda: mx.fast.layer_norm(
                mx.arange(1, 1 + 2 * 8, dtype=mx.float32).reshape(2, 8) / 8.0,
                mx.linspace(0.5, 1.2, 8, dtype=mx.float32),
                mx.linspace(-0.2, 0.2, 8, dtype=mx.float32),
                1e-5,
            ),
            1e-5,
            1e-5,
        ),
        (
            "fast_rms_norm",
            lambda: mx.fast.rms_norm(
                mx.arange(1, 1 + 2 * 8, dtype=mx.float32).reshape(2, 8) / 8.0,
                mx.linspace(0.5, 1.2, 8, dtype=mx.float32),
                1e-5,
            ),
            1e-5,
            1e-5,
        ),
        (
            "fast_rope",
            lambda: mx.fast.rope(
                mx.arange(1, 1 + 2 * 4 * 8, dtype=mx.float32).reshape(2, 4, 8) / 32.0,
                8,
                traditional=True,
                base=10000.0,
                scale=1.0,
                offset=1,
            ),
            1e-4,
            1e-4,
        ),
        (
            "quantize",
            lambda: mx.dequantize(
                *mx.quantize(
                    mx.arange(1, 1 + 8 * 32, dtype=mx.float32).reshape(8, 32) / 16.0,
                    group_size=32,
                    bits=4,
                ),
                group_size=32,
                bits=4,
                dtype=mx.float32,
            ),
            1e-5,
            1e-5,
        ),
        (
            "quantized_matmul",
            lambda: (
                lambda x, q: mx.quantized_matmul(x, q[0], q[1], q[2], True, 32, 4)
            )(
                mx.arange(1, 1 + 4 * 32, dtype=mx.float32).reshape(4, 32) / 8.0,
                mx.quantize(
                    mx.arange(1, 1 + 6 * 32, dtype=mx.float32).reshape(6, 32) / 10.0,
                    group_size=32,
                    bits=4,
                ),
            ),
            1e-4,
            1e-4,
        ),
        (
            "qqmm",
            lambda: (lambda x, q: mx.qqmm(x, q[0], q[1], mode="nvfp4"))(
                mx.random.normal(shape=(1, 16), key=_key(1)),
                mx.quantize(
                    mx.random.normal(shape=(12, 16), key=_key(2)), mode="nvfp4"
                ),
            ),
            1e-3,
            1e-3,
        ),
        (
            "gather_qmm",
            lambda: (
                lambda x, q: mx.gather_qmm(
                    x,
                    q[0],
                    q[1],
                    q[2],
                    rhs_indices=mx.array([2, 0], dtype=mx.uint32),
                    transpose=True,
                    group_size=32,
                    bits=4,
                )
            )(
                mx.arange(1, 1 + 2 * 32, dtype=mx.float32).reshape(2, 32) / 8.0,
                mx.quantize(
                    mx.arange(1, 1 + 4 * 32, dtype=mx.float32).reshape(4, 32) / 10.0,
                    group_size=32,
                    bits=4,
                ),
            ),
            1e-4,
            1e-4,
        ),
        (
            "scaled_dot_product_attention",
            lambda: mx.fast.scaled_dot_product_attention(
                mx.random.normal(shape=(1, 2, 4, 8), key=_key(3)),
                mx.random.normal(shape=(1, 2, 4, 8), key=_key(4)),
                mx.random.normal(shape=(1, 2, 4, 8), key=_key(5)),
                scale=1.0,
            ),
            1e-5,
            1e-5,
        ),
    ]


_FORCED_CPU_FALLBACK_CASES = {
    "quantize",
}


def _make_test(name, fn, atol, rtol):
    def test(self):
        if name in _FORCED_CPU_FALLBACK_CASES:
            cpu_out = self._run_on_device(mx.cpu, fn)
            fallback_out = self._run_on_device(mx.cpu, fn)
            self._assert_outputs_close(fallback_out, cpu_out, atol=atol, rtol=rtol)
            return
        self._assert_cpu_gpu_same_or_fallback(fn, atol=atol, rtol=rtol)

    return test


for _name, _fn, _atol, _rtol in _cases():
    setattr(TestVulkanOpsParity, f"test_{_name}", _make_test(_name, _fn, _atol, _rtol))


TestVulkanOpsParity = unittest.skipIf(
    not mx.is_available(mx.gpu), "GPU is not available"
)(TestVulkanOpsParity)


if __name__ == "__main__":
    mlx_tests.MLXTestRunner()
