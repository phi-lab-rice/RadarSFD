import torch
from diffusers import AutoencoderKL, AutoencoderTiny
import torch.nn as nn


class VAE:
    def __init__(self, device, torch_dtype, mode='tiny'):
        self.device = device
        self.torch_dtype = torch_dtype
        self.mode = mode
        self.scaling_factor = 0.18215
        self._init_vae()

    def _init_vae(self):
        if self.mode == 'tiny':
            model_path = "madebyollin/taesd"
            # When loading a model with known architecture mismatches,
            # it's safer to disable low_cpu_mem_usage. This prevents
            # the creation of "meta" tensors that cause errors on .to(device).
            vae = AutoencoderTiny.from_pretrained(
                model_path,
                torch_dtype=self.torch_dtype,
            ).to(self.device, memory_format=torch.channels_last)
            print("load TAESD")
        else:
            print("load SD VAE")
            model_path = "stabilityai/stable-diffusion-2-1"
            vae = AutoencoderKL.from_pretrained(
                model_path,
                subfolder="vae",
                torch_dtype=self.torch_dtype
            ).to(self.device, memory_format=torch.channels_last)

        for param in vae.parameters():
            param.requires_grad = False

        # Compile the entire VAE module after it's on the correct device
        self.vae = torch.compile(vae)

    def encode_latent(self, input_neg_one_to_one):
        """Encodes an input tensor from the range [-1, 1]."""
        input_tensor = input_neg_one_to_one.to(self.device, dtype=self.torch_dtype, memory_format=torch.channels_last)
        with ((torch.no_grad())):
            # Call the compiled VAE's encode method
            if self.mode == 'tiny':
                latents = self.vae.encode(input_tensor).latents * self.scaling_factor
            else:
                latents = self.vae.encode(input_tensor).latent_dist.sample() * self.scaling_factor

        return latents

    def decode_latent(self, latents):
        """Decodes latents back into image space."""
        latents = latents.to(self.device, dtype=self.torch_dtype, memory_format=torch.channels_last)
        # Keep the VAE frozen via requires_grad=False, but allow gradients
        # to flow through the decode graph back to the input latents.
        decoded = self.vae.decode(latents / self.scaling_factor).sample

        # Output in [0, 1] range
        decoded_zero_one = (decoded * 0.5 + 0.5).clamp(0, 1)
        # Output in [-1, 1] range
        decoded_neg_pos = decoded.clamp(-1, 1)

        return decoded_zero_one, decoded_neg_pos


if __name__ == '__main__':
    # Test lidar image reconstruction performance
    torch.random.manual_seed(0)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch_dtype = torch.float16 if device == "cuda" else torch.float32

    print("Initializing VAE...")
    # v = VAE(device, torch_dtype, mode='big')
    v = VAE(device, torch_dtype, mode='tiny')
    print("VAE initialized.")

    loss_fn = nn.MSELoss()

    # Create a dummy input tensor. VAEs expect input in the [-1, 1] range.
    # Let's create an input that is already normalized.
    input_neg_pos = 0.9 * torch.ones([4, 3, 256, 512]).clamp(-1, 1).to(device, dtype=torch_dtype)
    print(f"\nInput tensor shape: {input_neg_pos.shape}")
    print(f"Input tensor range: [{input_neg_pos.min():.2f}, {input_neg_pos.max():.2f}]")

    print("\nEncoding...")
    latents = v.encode_latent(input_neg_pos)
    print(f"Latents shape: {latents.shape}")

    print("Decoding...")
    decoded_zero_one, decoded_neg_pos = v.decode_latent(latents)
    print(f"Decoded [-1, 1] shape: {decoded_neg_pos.shape}")

    # Calculate reconstruction loss between the original [-1, 1] input and the decoded [-1, 1] output
    loss = loss_fn(decoded_neg_pos, input_neg_pos)
    print(f"\nReconstruction MSE Loss: {loss.item():.6f}")
