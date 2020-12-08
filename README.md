# pytorch-loss-functions

This repo is a combination of [victorca25/BasicSR](https://github.com/victorca25/BasicSR), [mit-han-lab/data-efficient-gans](https://github.com/mit-han-lab/data-efficient-gans) and [huster-wgm/Pytorch-metrics](https://github.com/huster-wgm/Pytorch-metrics/blob/master/metrics.py).

It aims to make the usage of different loss function, metrics and dataset augmentation easy and avoids using pip or other external depenencies.

Currently usable without major problems and with example usage in ```example.py```:
- [Differentiable Augmentation](https://github.com/mit-han-lab/data-efficient-gans)
- [Metrics (PSNR, SSIM, AE, MSE)](https://github.com/huster-wgm/Pytorch-metrics/blob/master/metrics.py)
- HFEN Loss (High Frequency Error Norm) [[1](https://ieeexplore.ieee.org/document/5617283) [2](https://www.hindawi.com/journals/cmmm/2016/7571934/)]
- Elastic Loss
- Relative L1
- L1 (CosineSim) [[1](https://github.com/dmarnerides/hdr-expandnet/blob/master/train.py) [2](https://arxiv.org/pdf/1803.02266.pdf)]
- Clip L1 [[1](https://github.com/HolmesShuan/AIM2020-Real-Super-Resolution/)]
- FFT Loss (Frequency loss) [[1](https://github.com/lj1995-computer-vision/Trident-Dehazing-Network/blob/master/loss/fft.py)]
- OF Loss (Overflow loss) [[1](https://github.com/lj1995-computer-vision/Trident-Dehazing-Network/blob/master/loss/brelu.py)]
- GP Loss (Gradient Profile Loss) [[1](https://github.com/ssarfraz/SPL/blob/master/SPL_Loss/)]
- CP Loss (Color Profile Loss)
- Style Loss [[1](https://github.com/Yukariin/DFNet/blob/master/loss.py) [2](https://github.com/pytorch/examples/blob/master/fast_neural_style/neural_style/utils.py)] (Warning: No AMP support.)
- TV Loss (Total Variation Loss)
- Perceptual Loss (LPIPS)
- Contextual Loss [[1](https://arxiv.org/abs/1803.02077) [2](https://github.com/roimehrez/contextualLoss) [3](https://github.com/S-aiueo32/contextual_loss_pytorch) [4](https://github.com/z-bingo/Contextual-Loss-PyTorch)]

May be added in the future in ```example.py```, but already in [```loss.py```](https://github.com/victorca25/BasicSR/tree/master/codes/models/modules):
- Charbonnier Loss (L1)
- GAN Loss
- Gradient Loss
- Masked L1 Loss
- Multiscale Pixel Loss
- OFR Loss (Optical flow reconstruction loss (for video)) [[1](https://github.com/LongguangWang/SOF-VSR/blob/master/TIP/data_utils.py)]
- L1 regularization
- Color Loss
- Average Loss (Averaging Downscale loss)
- SPL Compute With Trace (Spatial Profile Loss with trace)
- SP Loss (Spatial Profile Loss without trace)
