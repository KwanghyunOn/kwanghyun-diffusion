import torch
import torchvision

from guided_diffusion.script_util import create_model_and_diffusion, model_and_diffusion_defaults
from my_diffusion.sampling import sample_ddpm

if __name__ == "__main__":
    model_config = model_and_diffusion_defaults()
    model_config.update({
        'attention_resolutions': '32, 16, 8',
        'class_cond': False,
        'diffusion_steps': 1000,
        'dropout': 0.1,
        'image_size': 256,
        # 'rescale_timesteps': True,
        # 'timestep_respacing': '1000',
        'learn_sigma': True,
        'noise_schedule': 'linear',
        'num_channels': 256,
        'num_head_channels': 64,
        'num_res_blocks': 2,
        'resblock_updown': True,
        # 'use_checkpoint': False,
        'use_fp16': False,
        'use_scale_shift_norm': True,
    })

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print('Using device:', device)

    model, diffusion = create_model_and_diffusion(**model_config)
    model_path = "checkpoints/guided-diffusion/lsun_cat.pt"
    model.load_state_dict(torch.load(model_path, map_location='cpu'))
    model.requires_grad_(False).eval().to(device)

    x = torch.randn(1, 3, 256, 256).to(device)
    sample = sample_ddpm(x, model, torch.from_numpy(diffusion.betas))
    torchvision.utils.save_image((sample + 1.) * 0.5, "sample.png")    
    import pdb; pdb.set_trace()
