# File: rain_syn_enhanced.py (The Definitive 3-Module Orchestrator)

import torch
import yaml
from rainy.GuidedDisent.MUNIT.model_infer import MUNIT_infer
from rainy.GuidedDisent.droprenderer import DropModel
from torchvision import transforms
from PIL import Image
import os
from tqdm import tqdm
import numpy as np
import cv2
import argparse

# --- 1. IMPORT YOUR FINAL STREAK ENGINE ---
from rainy.GuidedDisent.rain_engine import generate_streak_layer

# --- UTILITY FUNCTION ---
def alpha_blend(background_bgr, foreground_rgba):
    """Performs alpha blending of a 4-channel RGBA foreground onto a 3-channel BGR background."""
    background_float = background_bgr.astype(float)
    fore_rgb = foreground_rgba[:, :, :3].astype(float)
    fore_alpha = foreground_rgba[:, :, 3].astype(float) / 255.0
    inv_alpha = 1.0 - fore_alpha
    blended_float = (fore_rgb * fore_alpha[..., np.newaxis]) + (background_float * inv_alpha[..., np.newaxis])
    return np.clip(blended_float, 0, 255).astype(np.uint8)

def load_image(image_path, resize=None):
    """Load image and convert to tensor format"""
    transform_list = []
    if resize:
        transform_list.append(transforms.Resize(resize))
    transform_list.append(transforms.ToTensor())
    transform = transforms.Compose(transform_list)
    image = Image.open(image_path).convert('RGB')
    return transform(image).unsqueeze(0)

class RainEffectProcessor:
    def __init__(self, config):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.munit_model = self._init_munit_model()
        self.drop_model = DropModel(imsize=self.config['imsize'])
        self.to_pil = transforms.ToPILImage()
        
    def _init_munit_model(self):
        """Initialize MUNIT model"""
        with open(self.config['params_path'], 'r') as stream:
            hyperparameters = yaml.load(stream, Loader=yaml.FullLoader)
        model = MUNIT_infer(hyperparameters)
        weights = torch.load(self.config['weights_path'])
        model.gen_a.load_state_dict(weights['a'])
        model.gen_b.load_state_dict(weights['b'])
        return model.to(self.device).eval()

    def process_image(self, img_tensor):
        """Processes an image with the definitive 3-stage pipeline."""
        
        # --- STAGE 1: Generate wetness 'mood' using MUNIT ---
        with torch.no_grad():
            im_wet_tensor = self.munit_model.forward(img_tensor)
        im_wet_np = (im_wet_tensor[0].cpu().permute(1, 2, 0).numpy() + 1) / 2.0 * 255.0
        im_wet_np_bgr = cv2.cvtColor(im_wet_np.astype(np.uint8), cv2.COLOR_RGB2BGR)

        # --- STAGE 2: Generate and composite falling streaks using your engine ---
        streak_layer_rgba = generate_streak_layer(
            image_shape=(im_wet_np_bgr.shape[0], im_wet_np_bgr.shape[1]),
            vehicle_speed_kmh=self.config['vehicle_speed_kmh'],
            rainfall_rate_mm_hr=self.config['rainfall_rate_mm_hr'],
            wind_speed_kmh=self.config['wind_speed_kmh']
        )
        im_with_streaks_bgr = alpha_blend(im_wet_np_bgr, streak_layer_rgba)

        # --- STAGE 3: Add refractive lens drops using original DropModel ---
        im_with_streaks_rgb = cv2.cvtColor(im_with_streaks_bgr, cv2.COLOR_BGR2RGB)
        im_with_streaks_tensor = (transforms.ToTensor()(im_with_streaks_rgb) * 2.0 - 1.0).unsqueeze(0).to(self.device)

        with torch.no_grad():
            drops_sigma = torch.zeros(1).fill_(self.config['drop_sigma']).to(self.device)
            im_final_tensor = self.drop_model.add_drops(im_with_streaks_tensor, sigma=drops_sigma)
        
        # --- Convert final results for saving ---
        im_final_pil = self.to_pil((im_final_tensor[0].cpu() + 1) / 2)
        
        return im_final_pil

def main():
    parser = argparse.ArgumentParser(description='Final Hybrid Rain Synthesis Pipeline')
    parser.add_argument('--input_dir', required=True, type=str, help='Input directory of clean images')
    parser.add_argument('--output_dir', required=True, type=str, help='Output directory for rainy images')
    parser.add_argument('--image_size', type=int, nargs=2, default=[512, 512], help='Image resolution')
    
    # Engine control parameters
    parser.add_argument('--vehicle_speed', type=float, default=30, help='Vehicle speed in km/h for perspective')
    parser.add_argument('--rainfall_rate', type=float, default=50, help='Rainfall rate in mm/hr for density')
    parser.add_argument('--wind_speed', type=float, default=0, help='Sideways wind in km/h for slant')

    # Original DropModel parameter
    parser.add_argument('--drop_sigma', type=float, default=4.0, help='Blur sigma for lens drops')

    args = parser.parse_args()
    
    config = {
        'imsize': tuple(args.image_size),
        'params_path': './rainy/GuidedDisent/configs/params_net.yaml',
        'weights_path': './rainy/GuidedDisent/weights/pretrained.pth',
        'vehicle_speed_kmh': args.vehicle_speed,
        'rainfall_rate_mm_hr': args.rainfall_rate,
        'wind_speed_kmh': args.wind_speed,
        'drop_sigma': args.drop_sigma
    }

    os.makedirs(args.output_dir, exist_ok=True)
    processor = RainEffectProcessor(config)
    image_files = sorted([f for f in os.listdir(args.input_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])

    for img_file in tqdm(image_files, desc="Processing images"):
        input_path = os.path.join(args.input_dir, img_file)
        img_tensor = load_image(input_path, resize=config['imsize']).to(processor.device)
        
        rainy_image = processor.process_image(img_tensor)
        
        output_path = os.path.join(args.output_dir, f'{os.path.splitext(img_file)[0]}_rainy.png')
        rainy_image.save(output_path)

if __name__ == '__main__':
    main()


# python rain_syn_enhanced.py --input_dir /scratch/Ananya_Kulkarni/JarvisIR/input --output_dir /scratch/Ananya_Kulkarni/JarvisIR/op_jarvis --vehicle_speed 50 --rainfall_rate 80