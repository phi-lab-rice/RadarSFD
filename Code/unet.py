import torch
from diffusers import UNet2DConditionModel


class marigoldUnet:
    def __init__(self, use_pretrained=True, device="cuda", torch_dtype=torch.float16):
        self.device = device
        self.torch_dtype = torch_dtype
        self.model_id = "prs-eth/marigold-depth-v1-1"
        self._init_unet(use_pretrained)

    def _init_unet(self, use_pretrained):
        if use_pretrained:
            unet = UNet2DConditionModel.from_pretrained(
                self.model_id, subfolder="unet"
            ).to(self.device, dtype=self.torch_dtype)

        else:
            config = UNet2DConditionModel.load_config(
                pretrained_model_name_or_path=self.model_id, subfolder="unet"
            )
            unet = UNet2DConditionModel.from_config(config).to(
                self.device, dtype=self.torch_dtype
            )

        self.unet = unet


class SDUnet:
    def __init__(self, use_pretrained=True, device="cuda", torch_dtype=torch.float16):
        self.device = device
        self.torch_dtype = torch_dtype
        self._init_unet(use_pretrained)

    def _init_unet(self, use_pretrained):
        model_id = "sd2-community/stable-diffusion-2-1"
        if use_pretrained:
            unet = UNet2DConditionModel.from_pretrained(model_id, subfolder="unet").to(
                self.device, dtype=self.torch_dtype
            )

        else:
            config = UNet2DConditionModel.load_config(
                pretrained_model_name_or_path=model_id, subfolder="unet"
            )
            unet = UNet2DConditionModel.from_config(config).to(
                self.device, dtype=self.torch_dtype
            )

        self.unet = unet
