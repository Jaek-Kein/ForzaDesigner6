"""OpenCL-based batch candidate scorer for the geometrize engine.

Provides `OpenCLScorer`, which evaluates N candidate shapes against the
current canvas on the GPU in a single kernel dispatch.  The caller
(GPUEngine) generates candidates on CPU, encodes them as a packed float
array, runs the kernel, and reads back (reg_new_sq, reg_old_sq, R, G, B)
per candidate.  Converting to RMS is done on the host so the kernel stays
simple and the precomputed canvas_full_sq is never sent to the GPU.

Supported shape types (GPU): circle, ellipse, rotated_ellipse,
                              rectangle, rotated_rectangle.
Unsupported (triangle): GPUEngine falls back to the CPU path.

Availability:
    opencl_available()  — quick check before constructing a scorer
    OpenCLScorer(...)   — raises RuntimeError if no usable GPU found
"""

from __future__ import annotations

import logging
import math
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from fd6.shapegen.shapes.base import Shape

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Optional pyopencl import
# ---------------------------------------------------------------------------
try:
    import pyopencl as cl
    _CL_OK = True
except ImportError:
    cl = None  # type: ignore[assignment]
    _CL_OK = False


def opencl_available() -> bool:
    """True if pyopencl is installed and at least one GPU device is present."""
    if not _CL_OK:
        return False
    try:
        for p in cl.get_platforms():
            if p.get_devices(cl.device_type.GPU):
                return True
    except Exception:
        pass
    return False


def _pick_device():
    """Return the GPU device with the most global memory, or None."""
    if not _CL_OK:
        return None
    best, best_mem = None, 0
    for p in cl.get_platforms():
        for d in p.get_devices(cl.device_type.GPU):
            try:
                mem = d.global_mem_size
            except Exception:
                mem = 0
            if mem > best_mem:
                best, best_mem = d, mem
    return best


# ---------------------------------------------------------------------------
# Shape encoding
# ---------------------------------------------------------------------------
# Supported type IDs used by the kernel.
_TYPE_IDS: dict[str, int] = {
    "circle": 0,
    "ellipse": 1,
    "rotated_ellipse": 2,
    "rectangle": 3,
    "rotated_rectangle": 4,
}
GPU_SUPPORTED_TYPES: frozenset[str] = frozenset(_TYPE_IDS)

# Packed layout: float32[N, 8]
#   [0] type_id
#   [1] cx         (x for circle/ellipse/rect; p0 for others)
#   [2] cy
#   [3] p0         circle→r; ellipse/rotated_ellipse→rx; rect/rotated_rect→hw
#   [4] p1         ellipse/rotated_ellipse→ry; rect/rotated_rect→hh; else 0
#   [5] p2         rotated types: angle in radians; else 0
#   [6] alpha      0-255
#   [7] _pad
_PARAMS = np.zeros(8, dtype=np.float32)


def encode_shapes(shapes: list["Shape"]) -> np.ndarray:
    """Return float32[N, 8] packed candidate buffer for the OpenCL kernel."""
    n = len(shapes)
    arr = np.zeros((n, 8), dtype=np.float32)
    for i, s in enumerate(shapes):
        tid = _TYPE_IDS.get(s.type_name, -1)
        if tid < 0:
            arr[i, 0] = -1.0  # unsupported sentinel — kernel must not see this
            continue
        arr[i, 0] = float(tid)
        arr[i, 6] = float(s.color[3])
        tn = s.type_name
        if tn == "circle":
            arr[i, 1] = s.x;  arr[i, 2] = s.y;  arr[i, 3] = s.r
        elif tn == "ellipse":
            arr[i, 1] = s.x;  arr[i, 2] = s.y
            arr[i, 3] = s.rx; arr[i, 4] = s.ry
        elif tn == "rotated_ellipse":
            arr[i, 1] = s.x;   arr[i, 2] = s.y
            arr[i, 3] = s.rx;  arr[i, 4] = s.ry
            arr[i, 5] = math.radians(s.angle)
        elif tn == "rectangle":
            arr[i, 1] = s.x;   arr[i, 2] = s.y
            arr[i, 3] = s.hw;  arr[i, 4] = s.hh
        elif tn == "rotated_rectangle":
            arr[i, 1] = s.x;   arr[i, 2] = s.y
            arr[i, 3] = s.hw;  arr[i, 4] = s.hh
            arr[i, 5] = math.radians(s.angle)
    return arr


# ---------------------------------------------------------------------------
# OpenCL kernel
# ---------------------------------------------------------------------------
# One work item per candidate.  Each item iterates its own bounding box in
# two passes:
#   Pass 1  — accumulate optimal RGB (closed-form colour), region_old_sq,
#             and sticker-overlap counts.
#   Pass 2  — compute region_new_sq using the optimal colour.
#
# The host then computes:
#   total_sq = canvas_full_sq - reg_old_sq + reg_new_sq
#   score    = sqrt(max(0, total_sq) / canvas_norm)
#
# Rejected candidates (sticker bleed, empty mask, empty bbox) set
# reg_new_sq = 1e30 so the host can detect inf reliably.

_KERNEL_SRC = r"""
#define SHAPE_CIRCLE            0
#define SHAPE_ELLIPSE           1
#define SHAPE_ROTATED_ELLIPSE   2
#define SHAPE_RECTANGLE         3
#define SHAPE_ROTATED_RECTANGLE 4

/* Candidate layout: float[8] per candidate — see encode_shapes() docstring. */
/* Result layout:   float[5] per candidate: [reg_new_sq, reg_old_sq, R, G, B] */

__kernel void score_shapes(
    __global const uchar* canvas,
    __global const uchar* target,
    __global const float* edge_weight,   /* H*W; ignored when has_ew==0 */
    __global const uchar* alpha_mask,    /* H*W; ignored when has_am==0 */
    __global const float* candidates,   /* N*8 */
    __global       float* results,      /* N*5 */
    const int W,
    const int H,
    const int has_ew,   /* 1 = use edge_weight buffer */
    const int has_am    /* 1 = sticker mode, use alpha_mask */
) {
    int ci   = get_global_id(0);
    int base = ci * 8;

    int   stype = (int)candidates[base + 0];
    float cx    = candidates[base + 1];
    float cy    = candidates[base + 2];
    float p0    = candidates[base + 3];
    float p1    = candidates[base + 4];
    float p2    = candidates[base + 5]; /* angle radians for rotated types */
    float a_val = candidates[base + 6]; /* alpha 0-255 */
    float a     = a_val / 255.0f;

    /* ---- bounding box ---- */
    int x0, y0, x1, y1;
    if (stype == SHAPE_CIRCLE) {
        x0 = max(0, (int)floor(cx - p0));
        y0 = max(0, (int)floor(cy - p0));
        x1 = min(W, (int)ceil(cx + p0 + 1.0f));
        y1 = min(H, (int)ceil(cy + p0 + 1.0f));
    } else if (stype == SHAPE_ELLIPSE) {
        x0 = max(0, (int)floor(cx - p0));
        y0 = max(0, (int)floor(cy - p1));
        x1 = min(W, (int)ceil(cx + p0 + 1.0f));
        y1 = min(H, (int)ceil(cy + p1 + 1.0f));
    } else if (stype == SHAPE_ROTATED_ELLIPSE) {
        float r = fmax(p0, p1);
        x0 = max(0, (int)floor(cx - r));
        y0 = max(0, (int)floor(cy - r));
        x1 = min(W, (int)ceil(cx + r + 1.0f));
        y1 = min(H, (int)ceil(cy + r + 1.0f));
    } else if (stype == SHAPE_RECTANGLE) {
        x0 = max(0, (int)floor(cx - p0));
        y0 = max(0, (int)floor(cy - p1));
        x1 = min(W, (int)ceil(cx + p0 + 1.0f));
        y1 = min(H, (int)ceil(cy + p1 + 1.0f));
    } else { /* ROTATED_RECTANGLE */
        float r = sqrt(p0 * p0 + p1 * p1);
        x0 = max(0, (int)floor(cx - r));
        y0 = max(0, (int)floor(cy - r));
        x1 = min(W, (int)ceil(cx + r + 1.0f));
        y1 = min(H, (int)ceil(cy + r + 1.0f));
    }

    int rb = ci * 5;
    if (x1 <= x0 || y1 <= y0) {
        results[rb+0] = 1e30f; results[rb+1] = 0.0f;
        results[rb+2] = 0.0f;  results[rb+3] = 0.0f; results[rb+4] = 0.0f;
        return;
    }

    float cos_a = 1.0f, sin_a = 0.0f;
    if (stype == SHAPE_ROTATED_ELLIPSE || stype == SHAPE_ROTATED_RECTANGLE) {
        cos_a = cos(p2); sin_a = sin(p2);
    }

    /* ---- pass 1: optimal colour + reg_old_sq + sticker check ---- */
    float sum_r = 0.0f, sum_g = 0.0f, sum_b = 0.0f, mw = 0.0f;
    float reg_old = 0.0f;
    float body_tot = 0.0f, body_in = 0.0f;

    for (int py = y0; py < y1; ++py) {
        for (int px = x0; px < x1; ++px) {
            float dx = (float)px - cx;
            float dy = (float)py - cy;

            float sm;
            if (stype == SHAPE_CIRCLE) {
                sm = (dx*dx + dy*dy <= p0*p0) ? 1.0f : 0.0f;
            } else if (stype == SHAPE_ELLIPSE) {
                float ndx = dx / fmax(p0, 1e-6f);
                float ndy = dy / fmax(p1, 1e-6f);
                sm = (ndx*ndx + ndy*ndy <= 1.0f) ? 1.0f : 0.0f;
            } else if (stype == SHAPE_ROTATED_ELLIPSE) {
                float xr = cos_a*dx + sin_a*dy;
                float yr = -sin_a*dx + cos_a*dy;
                float ndx = xr / fmax(p0, 1e-6f);
                float ndy = yr / fmax(p1, 1e-6f);
                sm = (ndx*ndx + ndy*ndy <= 1.0f) ? 1.0f : 0.0f;
            } else if (stype == SHAPE_RECTANGLE) {
                sm = (fabs(dx) <= p0 && fabs(dy) <= p1) ? 1.0f : 0.0f;
            } else {
                float xr = cos_a*dx + sin_a*dy;
                float yr = -sin_a*dx + cos_a*dy;
                sm = (fabs(xr) <= p0 && fabs(yr) <= p1) ? 1.0f : 0.0f;
            }

            /* sticker body counts use raw shape mask (not alpha-gated) */
            if (sm >= 0.5f) {
                body_tot += 1.0f;
                if (has_am) {
                    if (alpha_mask[py*W + px] >= 128) body_in += 1.0f;
                } else {
                    body_in += 1.0f;
                }
            }

            float em = sm;
            if (has_am) em = fmin(sm, (float)alpha_mask[py*W + px] / 255.0f);

            float ew = has_ew ? edge_weight[py*W + px] : 1.0f;
            int   i3 = (py*W + px) * 3;
            float cr = (float)canvas[i3],   cg = (float)canvas[i3+1], cb = (float)canvas[i3+2];
            float tr = (float)target[i3],   tg = (float)target[i3+1], tb = (float)target[i3+2];

            /* old error (ALL bbox pixels contribute) */
            float dr = cr-tr, dg = cg-tg, db = cb-tb;
            reg_old += ew * (dr*dr + dg*dg + db*db);

            /* optimal colour accumulation (effective-mask pixels only) */
            if (em > 1e-6f) {
                float inv_a = 1.0f / fmax(a, 1e-6f);
                sum_r += (tr - (1.0f-a)*cr) * inv_a * em;
                sum_g += (tg - (1.0f-a)*cg) * inv_a * em;
                sum_b += (tb - (1.0f-a)*cb) * inv_a * em;
                mw    += em;
            }
        }
    }

    /* sticker bleed rejection: >0.5% of body outside opaque region */
    if (has_am && body_tot > 0.5f && body_in / body_tot < 0.995f) {
        results[rb+0] = 1e30f; results[rb+1] = 0.0f;
        results[rb+2] = 0.0f;  results[rb+3] = 0.0f; results[rb+4] = 0.0f;
        return;
    }
    if (mw < 0.5f) {
        results[rb+0] = 1e30f; results[rb+1] = 0.0f;
        results[rb+2] = 0.0f;  results[rb+3] = 0.0f; results[rb+4] = 0.0f;
        return;
    }

    float opt_r = clamp(sum_r / mw, 0.0f, 255.0f);
    float opt_g = clamp(sum_g / mw, 0.0f, 255.0f);
    float opt_b = clamp(sum_b / mw, 0.0f, 255.0f);

    /* ---- pass 2: reg_new_sq with optimal colour (ALL bbox pixels) ---- */
    float reg_new = 0.0f;
    for (int py = y0; py < y1; ++py) {
        for (int px = x0; px < x1; ++px) {
            float dx = (float)px - cx;
            float dy = (float)py - cy;

            float sm;
            if (stype == SHAPE_CIRCLE) {
                sm = (dx*dx + dy*dy <= p0*p0) ? 1.0f : 0.0f;
            } else if (stype == SHAPE_ELLIPSE) {
                float ndx = dx / fmax(p0, 1e-6f);
                float ndy = dy / fmax(p1, 1e-6f);
                sm = (ndx*ndx + ndy*ndy <= 1.0f) ? 1.0f : 0.0f;
            } else if (stype == SHAPE_ROTATED_ELLIPSE) {
                float xr = cos_a*dx + sin_a*dy;
                float yr = -sin_a*dx + cos_a*dy;
                float ndx = xr / fmax(p0, 1e-6f);
                float ndy = yr / fmax(p1, 1e-6f);
                sm = (ndx*ndx + ndy*ndy <= 1.0f) ? 1.0f : 0.0f;
            } else if (stype == SHAPE_RECTANGLE) {
                sm = (fabs(dx) <= p0 && fabs(dy) <= p1) ? 1.0f : 0.0f;
            } else {
                float xr = cos_a*dx + sin_a*dy;
                float yr = -sin_a*dx + cos_a*dy;
                sm = (fabs(xr) <= p0 && fabs(yr) <= p1) ? 1.0f : 0.0f;
            }

            float em = sm;
            if (has_am) em = fmin(sm, (float)alpha_mask[py*W + px] / 255.0f);

            float ew = has_ew ? edge_weight[py*W + px] : 1.0f;
            int   i3 = (py*W + px) * 3;
            float cr = (float)canvas[i3],   cg = (float)canvas[i3+1], cb = (float)canvas[i3+2];
            float tr = (float)target[i3],   tg = (float)target[i3+1], tb = (float)target[i3+2];

            float br = em*(a*opt_r + (1.0f-a)*cr) + (1.0f-em)*cr;
            float bg = em*(a*opt_g + (1.0f-a)*cg) + (1.0f-em)*cg;
            float bb = em*(a*opt_b + (1.0f-a)*cb) + (1.0f-em)*cb;

            float er = br-tr, eg = bg-tg, eb = bb-tb;
            reg_new += ew * (er*er + eg*eg + eb*eb);
        }
    }

    results[rb+0] = reg_new;
    results[rb+1] = reg_old;
    results[rb+2] = opt_r;
    results[rb+3] = opt_g;
    results[rb+4] = opt_b;
}
"""


# ---------------------------------------------------------------------------
# OpenCLScorer
# ---------------------------------------------------------------------------

class OpenCLScorer:
    """GPU-side batch scorer.  Created once per generation; reused across iterations.

    Lifetime:
        scorer = OpenCLScorer(target, edge_weight, alpha_mask, w, h)
        for each iteration:
            results = scorer.score_batch(shapes, canvas, canvas_full_sq, canvas_norm)
        scorer.close()

    Raises RuntimeError if no OpenCL GPU is found or kernel compilation fails.
    """

    def __init__(
        self,
        target: np.ndarray,
        edge_weight: np.ndarray | None,
        alpha_mask: np.ndarray | None,
        w: int,
        h: int,
    ) -> None:
        if not _CL_OK:
            raise RuntimeError("pyopencl is not installed")
        dev = _pick_device()
        if dev is None:
            raise RuntimeError("No OpenCL GPU device found")

        self.device_name: str = dev.name.strip()
        self._w = w
        self._h = h
        self._has_ew = edge_weight is not None
        self._has_am = alpha_mask is not None

        self._ctx = cl.Context([dev])
        self._queue = cl.CommandQueue(self._ctx)

        # Compile kernel (JIT, one-time cost per session).
        self._prog = cl.Program(self._ctx, _KERNEL_SRC).build()

        mf = cl.mem_flags
        canvas_bytes = h * w * 3

        # Static read-only buffers uploaded once.
        self._target_buf = cl.Buffer(
            self._ctx, mf.READ_ONLY | mf.COPY_HOST_PTR,
            hostbuf=np.ascontiguousarray(target, dtype=np.uint8),
        )
        if edge_weight is not None:
            self._ew_buf = cl.Buffer(
                self._ctx, mf.READ_ONLY | mf.COPY_HOST_PTR,
                hostbuf=np.ascontiguousarray(edge_weight, dtype=np.float32),
            )
        else:
            self._ew_buf = cl.Buffer(self._ctx, mf.READ_ONLY, size=4)  # dummy

        if alpha_mask is not None:
            self._am_buf = cl.Buffer(
                self._ctx, mf.READ_ONLY | mf.COPY_HOST_PTR,
                hostbuf=np.ascontiguousarray(alpha_mask, dtype=np.uint8),
            )
        else:
            self._am_buf = cl.Buffer(self._ctx, mf.READ_ONLY, size=1)  # dummy

        # Canvas buffer: pre-allocated, content re-uploaded every iteration.
        self._canvas_buf = cl.Buffer(self._ctx, mf.READ_ONLY, size=canvas_bytes)

        # Per-batch buffers (lazy-allocated / reused when size is stable).
        self._batch_n: int = 0
        self._cands_buf: cl.Buffer | None = None
        self._results_buf: cl.Buffer | None = None
        self._results_host: np.ndarray = np.empty(0, dtype=np.float32)

    # ------------------------------------------------------------------
    def update_edge_weight(self, edge_weight: np.ndarray) -> None:
        """Re-upload edge weight map (called when residual refresh is enabled)."""
        arr = np.ascontiguousarray(edge_weight, dtype=np.float32)
        cl.enqueue_copy(self._queue, self._ew_buf, arr)

    # ------------------------------------------------------------------
    def _ensure_batch_bufs(self, n: int) -> None:
        if n == self._batch_n:
            return
        mf = cl.mem_flags
        self._cands_buf = cl.Buffer(self._ctx, mf.READ_ONLY,  size=n * 8 * 4)
        self._results_buf = cl.Buffer(self._ctx, mf.WRITE_ONLY, size=n * 5 * 4)
        self._results_host = np.empty((n, 5), dtype=np.float32)
        self._batch_n = n

    # ------------------------------------------------------------------
    def score_batch(
        self,
        shapes: list["Shape"],
        canvas: np.ndarray,
        canvas_full_sq: float,
        canvas_norm: float,
    ) -> list[tuple[float, int, int, int]]:
        """Score all shapes against the given canvas snapshot.

        Returns a list of (rms_score, opt_r, opt_g, opt_b).
        Rejected candidates return (inf, 0, 0, 0).
        """
        n = len(shapes)
        if n == 0:
            return []

        # Upload current canvas (changes every iteration).
        canvas_c = np.ascontiguousarray(canvas, dtype=np.uint8)
        cl.enqueue_copy(self._queue, self._canvas_buf, canvas_c)

        # Encode candidates and upload.
        cands = encode_shapes(shapes)
        self._ensure_batch_bufs(n)
        cl.enqueue_copy(self._queue, self._cands_buf, cands)

        # Dispatch kernel: 1 work item per candidate.
        self._prog.score_shapes(
            self._queue, (n,), None,
            self._canvas_buf,
            self._target_buf,
            self._ew_buf,
            self._am_buf,
            self._cands_buf,
            self._results_buf,
            np.int32(self._w),
            np.int32(self._h),
            np.int32(int(self._has_ew)),
            np.int32(int(self._has_am)),
        )

        # Download results and wait for completion.
        cl.enqueue_copy(self._queue, self._results_host, self._results_buf)
        self._queue.finish()

        # Convert (reg_new_sq, reg_old_sq) → RMS on host.
        out: list[tuple[float, int, int, int]] = []
        for i in range(n):
            reg_new, reg_old, r, g, b = self._results_host[i]
            if reg_new >= 1e29:
                out.append((float("inf"), 0, 0, 0))
            else:
                total_sq = canvas_full_sq - float(reg_old) + float(reg_new)
                score = (
                    math.sqrt(max(0.0, total_sq) / canvas_norm)
                    if canvas_norm > 0
                    else 0.0
                )
                out.append((score, int(round(r)), int(round(g)), int(round(b))))
        return out

    # ------------------------------------------------------------------
    def close(self) -> None:
        """Release GPU resources.  Safe to call multiple times."""
        for attr in ("_canvas_buf", "_target_buf", "_ew_buf", "_am_buf",
                     "_cands_buf", "_results_buf"):
            buf = getattr(self, attr, None)
            if buf is not None:
                try:
                    buf.release()
                except Exception:
                    pass
                setattr(self, attr, None)
        self._batch_n = 0

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass
