"""GPU-accelerated geometrize engine.

`GPUEngine` is a drop-in replacement for `Engine` that overrides
`_parallel_search` to dispatch batch candidate scoring to an OpenCL GPU.
The CPU path (ProcessPoolExecutor) is kept as a fallback for:
  - pyopencl not installed / no GPU device
  - shape types not supported on GPU (triangle)
  - any runtime OpenCL error during scoring

Algorithm per iteration (GPU path):
  1. Generate N_CHAINS × random_samples candidate shapes on CPU.
  2. Upload canvas + all candidates to GPU.
  3. Run the score_shapes kernel — one work item per candidate.
  4. Download scores; find the top-K starters.
  5. Hill-climb each starter on CPU (sequential); return the global best.

Step 5 is deliberately kept on CPU: hill-climbing is inherently sequential
and the 200-step climb is cheap compared to the 1000+ random evaluations
in step 3.

`create_engine(...)` is the public factory.  It tries to build a GPUEngine;
if OpenCL is unavailable it falls back to the plain Engine transparently.
"""

from __future__ import annotations

import logging
import math

import numpy as np

from fd6.shapegen.engine import Engine, EngineConfig
from fd6.shapegen.scoring import precompute_canvas_error, score_shape
from fd6.shapegen.shapes.base import Shape, random_shape
from fd6.shapegen.gpu_search import (
    GPU_SUPPORTED_TYPES,
    OpenCLScorer,
    opencl_available,
)

logger = logging.getLogger(__name__)

# Number of independent random-search chains run through the GPU each
# iteration.  Equivalent to `n_workers` in the CPU path — the GPU batches
# all chains together instead of running them in separate processes.
_GPU_CHAINS = 4  # good balance of diversity vs kernel dispatch overhead


class GPUEngine(Engine):
    """Engine with OpenCL-accelerated random-search phase.

    Falls back to the parent CPU path (ProcessPoolExecutor) when:
      - scorer is None (init failed)
      - current shape type is not GPU-supported (e.g., triangle)
    """

    def __init__(
        self,
        target_rgb: np.ndarray,
        config: EngineConfig,
        alpha_mask: np.ndarray | None = None,
    ) -> None:
        super().__init__(target_rgb, config, alpha_mask)
        self._scorer: OpenCLScorer | None = None

        if not opencl_available():
            logger.debug("GPUEngine: no OpenCL GPU found — CPU fallback active")
            return

        try:
            self._scorer = OpenCLScorer(
                target=self.target,
                # Use the base (immutable) edge-weight map, not the shared-
                # memory live view, since we capture it once at init time.
                edge_weight=self._base_edge_weight,
                alpha_mask=self.alpha_mask,
                w=self.w,
                h=self.h,
            )
            logger.info("GPUEngine: OpenCL active — device: %s", self._scorer.device_name)
        except Exception as exc:
            logger.warning("GPUEngine: OpenCL init failed (%s) — CPU fallback active", exc)
            self._scorer = None

    # ------------------------------------------------------------------
    # Residual reblend hook (currently disabled by RESIDUAL_REFRESH_EVERY=0)
    # ------------------------------------------------------------------
    def _refresh_residual_weight(self) -> None:
        super()._refresh_residual_weight()
        if self._scorer is not None:
            self._scorer.update_edge_weight(self.edge_weight)

    # ------------------------------------------------------------------
    # Core override
    # ------------------------------------------------------------------
    def _parallel_search(
        self,
        types: list[str],
        n_random: int,
        n_mutate: int,
        max_size_frac: float | None = None,
    ) -> tuple[float, Shape | None]:
        """GPU batch random-search + CPU hill-climb.

        Falls back to the parent ProcessPoolExecutor path when the scorer
        is unavailable or when any requested shape type lacks GPU support.
        """
        if self._scorer is None or not GPU_SUPPORTED_TYPES.issuperset(types):
            return super()._parallel_search(types, n_random, n_mutate, max_size_frac)

        n_random = max(1, n_random)
        n_mutate = max(1, n_mutate)
        n_chains = max(_GPU_CHAINS, self._n_workers)
        n_total  = n_random * n_chains

        # --- generate all candidates on CPU ---
        shapes = [
            random_shape(self.rng, self.w, self.h, types, max_size_frac)
            for _ in range(n_total)
        ]

        # --- precompute canvas error (fast CPU numpy, done once) ---
        canvas_full_sq, canvas_norm = precompute_canvas_error(
            self.canvas, self.target, self.alpha_mask, self.edge_weight,
        )
        if canvas_norm < 1:
            return float("inf"), None

        # --- GPU batch scoring ---
        try:
            gpu_results = self._scorer.score_batch(
                shapes, self.canvas, canvas_full_sq, canvas_norm,
            )
        except Exception as exc:
            logger.warning("GPUEngine: score_batch failed (%s) — using CPU fallback", exc)
            self._scorer = None
            return super()._parallel_search(types, n_random, n_mutate, max_size_frac)

        # --- pick top-K starters for hill climbing ---
        scores = [r[0] for r in gpu_results]
        # Number of independent hill-climb restarts (≤ n_chains, capped at 4
        # to keep the CPU hill-climb phase cheap).
        n_restart = min(4, n_chains)
        top_indices = sorted(range(n_total), key=lambda i: scores[i])[:n_restart]

        best_score = float("inf")
        best_shape: Shape | None = None

        for idx in top_indices:
            score, r, g, b = gpu_results[idx]
            if score == float("inf"):
                continue
            starter = shapes[idx]
            alpha   = starter.color[3]
            starter.color = (r, g, b, alpha)

            result_shape, result_score = self._hill_climb(
                starter, score, n_mutate, canvas_full_sq, canvas_norm,
            )
            if result_score < best_score:
                best_score = result_score
                best_shape = result_shape

        return best_score, best_shape

    # ------------------------------------------------------------------
    def _hill_climb(
        self,
        shape: Shape,
        best_score: float,
        n_mutate: int,
        canvas_full_sq: float,
        canvas_norm: float,
    ) -> tuple[Shape, float]:
        """Single-chain hill climb on CPU starting from `shape`."""
        cap = max(1, n_mutate)
        no_improve = 0
        for _ in range(cap):
            cand = shape.mutate(self.rng, self.w, self.h)
            score, color = score_shape(
                cand, self.canvas, self.target, self.alpha_mask,
                canvas_full_sq=canvas_full_sq,
                canvas_norm=canvas_norm,
                edge_weight=self.edge_weight,
            )
            if score < best_score:
                best_score = score
                shape = cand
                shape.color = color
                no_improve = 0
            else:
                no_improve += 1
                if no_improve >= max(20, cap // 4):
                    break
        return shape, best_score

    # ------------------------------------------------------------------
    def _shutdown(self) -> None:
        if self._scorer is not None:
            try:
                self._scorer.close()
            except Exception:
                pass
            self._scorer = None
        super()._shutdown()


# ---------------------------------------------------------------------------
# Public factory
# ---------------------------------------------------------------------------

def create_engine(
    target_rgb: np.ndarray,
    config: EngineConfig,
    alpha_mask: np.ndarray | None = None,
) -> Engine:
    """Return a GPUEngine if OpenCL is available, otherwise a plain Engine.

    GPUEngine itself falls back to CPU silently if scorer init fails at
    runtime, so this factory only avoids the overhead of constructing
    shared-memory worker pools when OpenCL is definitely absent.
    """
    if opencl_available():
        try:
            return GPUEngine(target_rgb, config, alpha_mask)
        except Exception as exc:
            logger.warning("create_engine: GPUEngine constructor failed (%s) — using CPU", exc)
    return Engine(target_rgb, config, alpha_mask)
