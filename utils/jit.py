"""numba as an optional accelerator: ``njit``/``prange`` fall back to no-ops without it."""

try:
    from numba import njit, prange
    HAVE_NUMBA = True
except ImportError:  # pragma: no cover - numba is an optional accelerator
    HAVE_NUMBA = False

    def njit(func=None, **_):  # type: ignore[misc]
        return func if func is not None else (lambda f: f)

    prange = range  # type: ignore[assignment]

__all__ = ["HAVE_NUMBA", "njit", "prange"]
