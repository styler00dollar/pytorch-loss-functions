"""
This is just an example and should show basic usage
Write proper syntax yourself / Add it to existing code

Examples with working code:
DFNet:
https://github.com/styler00dollar/Colab-DFNet

deepfillv2:
https://github.com/styler00dollar/Colab-mmediting/tree/master/mmedit/models/inpaintors
"""

from vic.loss import CharbonnierLoss, GANLoss, GradientPenaltyLoss, HFENLoss, TVLoss, GradientLoss, ElasticLoss, RelativeL1, L1CosineSim, ClipL1, MaskedL1Loss, MultiscalePixelLoss, FFTloss, OFLoss, L1_regularization, ColorLoss, AverageLoss, GPLoss, CPLoss, SPL_ComputeWithTrace, SPLoss, Contextual_Loss, StyleLoss
from vic.perceptual_loss import PerceptualLoss
from vic.filters import *
from vic.colors import *
from vic.discriminators import *
from .diffaug import *

from metrics import *

from tensorboardX import SummaryWriter

logdir='/path/'

writer = SummaryWriter(logdir=logdir)

class CalcLoss(nn.Module):
    def __init__(self):
        super().__init__()

        """
        if self.config.HFEN_TYPE == 'L1':
          l_hfen_type = nn.L1Loss()
        if self.config.HFEN_TYPE == 'MSE':
          l_hfen_type = nn.MSELoss()
        if self.config.HFEN_TYPE == 'Charbonnier':
          l_hfen_type = CharbonnierLoss()
        if self.config.HFEN_TYPE == 'ElasticLoss':
          l_hfen_type = ElasticLoss()
        if self.config.HFEN_TYPE == 'RelativeL1':
          l_hfen_type = RelativeL1()
        if self.config.HFEN_TYPE == 'L1CosineSim':
          l_hfen_type = L1CosineSim()
        """

        l_hfen_type = L1CosineSim()
        self.HFENLoss = HFENLoss(loss_f=l_hfen_type, kernel='log', kernel_size=15, sigma = 2.5, norm = False)

        self.ElasticLoss = ElasticLoss(a=0.2, reduction='mean')

        self.RelativeL1 = RelativeL1(eps=.01, reduction='mean')

        self.L1CosineSim = L1CosineSim(loss_lambda=5, reduction='mean')

        self.ClipL1 = ClipL1(clip_min=0.0, clip_max=10.0)

        self.FFTloss = FFTloss(loss_f = torch.nn.L1Loss, reduction='mean')

        self.OFLoss = OFLoss()

        self.GPLoss = GPLoss(trace=False, spl_denorm=False)

        self.CPLoss = CPLoss(rgb=True, yuv=True, yuvgrad=True, trace=False, spl_denorm=False, yuv_denorm=False)

        self.StyleLoss = StyleLoss() # Warning: does not support AMP

        self.TVLoss = TVLoss(tv_type='tv', p = 1)

        self.PerceptualLoss = PerceptualLoss(model='net-lin', net='alex', colorspace='rgb', spatial=False, use_gpu=True, gpu_ids=[0], model_path=None)

        layers_weights = {'conv_1_1': 1.0, 'conv_3_2': 1.0}
        self.Contextual_Loss = Contextual_Loss(layers_weights, crop_quarter=False, max_1d_size=100,
            distance_type = 'cosine', b=1.0, band_width=0.5,
            use_vgg = True, net = 'vgg19', calc_type = 'regular')

        self.psnr_metric = PSNR()
        self.ssim_metric = SSIM()
        self.ae_metric = AE()
        self.mse_metric = MSE()


        def forward(self, input, gt, iteration):
            # out is generated output, gt_res is original image

            # loss functions
            # You can also add them to tensorboard with writer.add_scalar
            HFENLoss_forward = self.HFENLoss(out, gt_res)

            ElasticLoss_forward = self.ElasticLoss(out, gt_res)

            RelativeL1_forward = self.RelativeL1(out, gt_res)

            L1CosineSim_forward += 6*self.L1CosineSim(out, gt_res)

            ClipL1_forward = self.ClipL1(out, gt_res)

            FFTloss_forward = self.FFTloss(out, gt_res)

            OFLoss_forward = self.OFLoss(out)

            GPLoss_forward = self.GPLoss(out, gt_res)

            CPLoss_forward = self.CPLoss(out, gt_res)

            Contextual_Loss_forward = self.Contextual_Loss(out, gt_res)

            style_forward = self.StyleLoss(out, gt_res) # Warning: does not support AMP

            tv_forward = self.TVLoss(out)

            total_loss = tv_forward

            perceptual_forward = self.PerceptualLoss(out, gt_res)


            # improving GANloss by adding DiffAug
            # Discriminator loop
            out = Discriminator(DiffAugment(out, policy=policy))
            gt_res = Discriminator(DiffAugment(gt_res, policy=policy))
            # calc GANloss

            # Generator loop
            out = Discriminator(DiffAugment(out, policy=policy))
            # calc GANloss


            # writing metrics to tensorboard file
            # PSNR (Peak Signal-to-Noise Ratio)
            writer.add_scalar('metrics/PSNR', self.psnr_metric(gt_res, out), iteration)

            # SSIM (Structural Similarity)
            writer.add_scalar('metrics/SSIM', self.ssim_metric(gt_res, out), iteration)

            # AE (Average Angular Error)
            writer.add_scalar('metrics/AE', self.ae_metric(gt_res, out), iteration)

            # MSE (Mean Square Error)
            writer.add_scalar('metrics/MSE', self.mse_metric(gt_res, out), iteration)
