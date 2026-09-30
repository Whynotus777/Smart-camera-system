"""Interface for anything that yields frames: RTSP cameras, files, sim output, emulated streams.

Every downstream stage consumes this one Protocol, so a pipeline can be pointed at a
live camera, a recorded clip, Isaac Sim frames, or camera-emulated footage without
code changes. Signature is fixed by docs/ARCHITECTURE.md §4; change it only via ADR.
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import TYPE_CHECKING, Protocol, runtime_checkable

import numpy as np

from scs.contracts import FrameRef

if TYPE_CHECKING:
    import torch


@runtime_checkable
class FrameSource(Protocol):
    """A stream of decoded frames from one camera (T02)."""

    def frames(self) -> Iterator[tuple[FrameRef, np.ndarray | torch.Tensor]]:
        """Yield `(ref, image)` pairs in decode order.

        `image` is a torch tensor on the GPU when decoded with NVDEC, otherwise an
        HxWx3 numpy array. `ref.width`/`ref.height` describe `image`.
        """
        ...
