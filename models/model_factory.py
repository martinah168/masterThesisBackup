import sys

from models.unet_with_encoder import Diffusion_Autoencoder_Model

sys.path.append("..")
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from utils.arguments import DAE_Option
else:
    DAE_Option = object()
import pandas as pd
from monai.networks.nets.densenet import densenet121
from enum import Enum, auto
from torch.nn import Module
from torch import nn
from torch import tanh
import torch
from monai.networks.nets.vit import ViT

from models import Model
from utils.enums_model import ModelName, ModelType, LatentNetType


def get_model(opt: DAE_Option) -> Model:
    img_size = opt.img_size
    if isinstance(img_size, list):
        if len(img_size) == 1:
            img_size = img_size[0]

    assert isinstance(img_size, int), img_size
    # if isinstance(int) else opt.
    if opt.model_name in [ModelName.autoencoder]:
        if opt.model_name == ModelName.autoencoder:
            opt.model_type = ModelType.autoencoder
        else:
            raise NotImplementedError()

        if opt.net_latent_net_type == LatentNetType.none:
            latent_net_conf = None
        elif opt.net_latent_net_type == LatentNetType.skip:
            raise NotImplementedError(opt.net_latent_net_type)
            # latent_net_conf = MLPSkipNetConfig(
            #    num_channels=opt.style_ch,
            #    skip_layers=opt.net_latent_skip_layers,
            #    num_hid_channels=opt.net_latent_num_hid_channels,
            #    num_layers=opt.net_latent_layers,
            #    num_time_emb_channels=opt.net_latent_time_emb_channels,
            #    activation=opt.net_latent_activation,
            #    use_norm=opt.net_latent_use_norm,
            #    condition_bias=opt.net_latent_condition_bias,
            #    dropout=opt.net_latent_dropout,
            #    last_act=opt.net_latent_net_last_act,
            #    num_time_layers=opt.net_latent_num_time_layers,
            #    time_last_act=opt.net_latent_time_last_act,
            # )
        else:
            raise NotImplementedError(opt.net_latent_net_type)

        return Diffusion_Autoencoder_Model(
            attention_resolutions=opt.attention_resolutions,
            channel_mult=opt.net_ch_mult,
            conv_resample=True,
            dims=opt.dims,
            dropout=opt.dropout,
            embed_channels=opt.embed_channels,
            enc_out_channels=opt.enc_out_channels,
            enc_pool=opt.net_enc_pool,
            enc_num_res_block=opt.enc_num_res_blocks,
            enc_channel_mult=opt.enc_channel_mult,
            enc_grad_checkpoint=opt.enc_grad_checkpoint,
            enc_attn_resolutions=opt.enc_attn,
            image_size=img_size,
            in_channels=opt.in_channels,
            model_channels=opt.model_channels,
            num_classes=None,
            num_head_channels=-1,
            num_heads_upsample=-1,
            num_heads=opt.net_beatgans_attn_head,
            num_res_blocks=opt.net_num_res_blocks,
            num_input_res_blocks=opt.net_num_input_res_blocks,
            out_channels=opt.model_out_channels,
            resblock_updown=opt.net_resblock_updown,
            use_checkpoint=opt.net_beatgans_gradient_checkpoint,
            use_new_attention_order=False,
            resnet_two_cond=opt.net_beatgans_resnet_two_cond,
            resnet_use_zero_module=opt.net_beatgans_resnet_use_zero_module,
            latent_net=latent_net_conf,
            resnet_cond_channels=opt.net_beatgans_resnet_cond_channels,
        )
    else:
        raise NotImplementedError(opt.model_name)

    return opt.model_conf
