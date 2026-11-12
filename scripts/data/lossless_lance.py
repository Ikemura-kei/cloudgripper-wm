"""A Lance writer that stores frames losslessly.

``stable_worldmodel``'s :class:`LanceWriter` JPEG-encodes every image column.
For photographic scenes that is the right trade, but it is fatal for the flat
2D scenes (``flat2d/Ball-v0``), whose whole premise is that a frame contains
exactly three RGB triples: measured on one 448x448 flat2d frame, the default
quality 95 reads back as 449 distinct colours with errors up to 87/255 of
ringing around the ball's edge, and even quality 100 leaves 115 colours. PNG
round-trips byte-exact and, on frames this flat, is also ~3.8x smaller.

Nothing on the read side needs to change: ``LanceDataset`` tries
``torchvision.io.decode_jpeg`` first, which raises ``RuntimeError`` on a PNG
blob, and already falls back to a PIL loop that decodes PNG fine. The dataset
is an ordinary Lance table in every other respect, so resume-counting, config
checks and ``swm.data.load_dataset`` all work unchanged.
"""

from __future__ import annotations

import io

import numpy as np
import pyarrow as pa
from PIL import Image
from stable_worldmodel.data.formats.lance import LanceWriter


def _encode_frame_png(frame: np.ndarray) -> bytes:
    """PNG-encodes one frame, matching LanceWriter's channel handling.

    args:
        frame: uint8 image, (H, W, C) or channels-first (C, H, W)
    returns:
        PNG bytes
    """
    if frame.ndim == 3 and frame.shape[0] in (1, 3, 4):
        frame = np.transpose(frame, (1, 2, 0))
    if frame.shape[-1] == 1:
        frame = frame.squeeze(-1)
    buf = io.BytesIO()
    Image.fromarray(frame.astype(np.uint8)).save(buf, format='PNG')
    return buf.getvalue()


class LosslessLanceWriter(LanceWriter):
    """:class:`LanceWriter` with PNG image blobs instead of JPEG."""

    def _build_batch(self, ep_data: dict, ep_len: int) -> pa.RecordBatch:
        """Mirrors ``LanceWriter._build_batch`` with PNG-encoded image columns.

        The parent reaches for a module-level ``_encode_frame`` rather than an
        overridable hook, so the body is restated here rather than patched —
        swapping that global would race with the background thread lancedb
        pulls batches from. Non-image columns are built exactly as upstream
        does, so the resulting schema is identical.

        args:
            ep_data: one episode's columns, each a sequence of per-step values
            ep_len: number of steps in the episode
        returns:
            a record batch matching ``self._schema``
        """
        arrays: list[pa.Array] = [
            pa.array(np.full(ep_len, self._ep_idx, dtype=np.int32), type=pa.int32()),
            pa.array(np.arange(ep_len, dtype=np.int32), type=pa.int32()),
        ]
        for col, lance_name in self._rename_map.items():
            vals = ep_data[col]
            if lance_name in self._image_cols:
                blobs = [_encode_frame_png(np.asarray(v)) for v in vals]
                arrays.append(pa.array(blobs, type=pa.binary()))
            else:
                dim = self._dims[lance_name]
                flat = np.asarray(vals, dtype=np.float32).reshape(ep_len, dim)
                arrays.append(
                    pa.FixedSizeListArray.from_arrays(
                        pa.array(flat.reshape(-1), type=pa.float32()), dim
                    )
                )

        return pa.record_batch(arrays, schema=self._schema)
