"""Postprocessing"""

import torch

from .abc import Stage


def _postprocess(
    mask: torch.Tensor, noisy_in: torch.Tensor, win: torch.Tensor, device: str
) -> torch.Tensor:
    """Postprocessing function

    1. Applying mask to noisy signal.
    2. Filling 257-th bin of spectrogram with zeros
    3. Performs iSTFT

    Returns a wav tensor
    """
    # Applying mask
    denoised_spec_256 = noisy_in * mask

    # Filling 257-th bin with zeros
    pad = torch.zeros(
        (denoised_spec_256.shape[0], 1, 1, denoised_spec_256.shape[-1]),
        dtype=denoised_spec_256.dtype,
        device=device,
    )
    denoised_spec_257 = torch.cat([denoised_spec_256, pad], dim=2)

    # Inverse STFT
    denoised_spec_257 = denoised_spec_257.squeeze(1)
    denoised_wav = torch.istft(  # pylint: disable=E1102
        denoised_spec_257, n_fft=512, hop_length=160, window=win
    )

    return denoised_wav


class PostprocessStage(Stage[torch.Tensor, torch.Tensor]):
    """Postprocessing stage
    1. Applying mask to noisy signal.
    2. Filling 257-th bin of spectrogram with zeros
    3. Performs iSTFT
    """

    def __init__(self, noisy_in: torch.Tensor, win: torch.Tensor, device: str):
        self.noisy_in = noisy_in
        self.win = win
        self.device = device

        super().__init__(self._run)

    def _run(self, mask: torch.Tensor) -> torch.Tensor:
        return _postprocess(mask, self.noisy_in, self.win, self.device)
