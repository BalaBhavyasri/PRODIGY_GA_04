import os
import argparse
import torch
import torchvision.transforms as transforms
from PIL import Image
from torchvision.utils import save_image
import torchvision.transforms.functional as TF
from models import UNetGenerator
def load_generator(checkpoint_path, device):
    """
    Instantiate UNetGenerator and load weight state dict from checkpoint.
    """
    generator = UNetGenerator(input_nc=3, output_nc=3).to(device)
    
    # Load state dict
    checkpoint = torch.load(checkpoint_path, map_location=device)
    if 'generator_state_dict' in checkpoint:
        generator.load_state_dict(checkpoint['generator_state_dict'])
    else:
        generator.load_state_dict(checkpoint) # fallback if only state_dict was saved
        
    generator.eval()
    return generator
def preprocess_image(image_path, target_size=256):
    """
    Load and preprocess input image.
    """
    image = Image.open(image_path).convert('RGB')
    
    # Standard translation transforms (resize and normalize to [-1, 1])
    transform = transforms.Compose([
        transforms.Resize((target_size, target_size), Image.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    
    # Add batch dimension: [3, 256, 256] -> [1, 3, 256, 256]
    tensor = transform(image).unsqueeze(0)
    return tensor
def main():
    parser = argparse.ArgumentParser(description="Generate image-to-image translation using a trained Pix2Pix model")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to trained generator checkpoint (.pth file)")
    parser.add_argument("--input", type=str, required=True, help="Path to input image or directory of images to translate")
    parser.add_argument("--output_dir", type=str, default="results", help="Directory where results will be saved")
    parser.add_argument("--target_size", type=int, default=256, help="Target image size for model input")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Device to use ('cuda' or 'cpu')")
    parser.add_argument("--save_comparison", action="store_true", default=True, help="Save a side-by-side comparison of Input and Output")
    parser.add_argument("--no_comparison", action="store_false", dest="save_comparison", help="Do not save a side-by-side comparison")
    
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load model
    print(f"Loading model checkpoint from {args.checkpoint}...")
    device = torch.device(args.device)
    generator = load_generator(args.checkpoint, device)
    
    # Gather input paths
    if os.path.isdir(args.input):
        input_files = [
            os.path.join(args.input, f) for f in os.listdir(args.input)
            if f.lower().endswith(('.png', '.jpg', '.jpeg'))
        ]
        print(f"Found {len(input_files)} images in directory {args.input}")
    else:
        input_files = [args.input]
        
    # Process images
    with torch.no_grad():
        for img_path in input_files:
            filename = os.path.basename(img_path)
            name, ext = os.path.splitext(filename)
            
            print(f"Translating image: {filename}...")
            
            # Load and preprocess
            input_tensor = preprocess_image(img_path, args.target_size).to(device)
            
            # Forward pass
            output_tensor = generator(input_tensor)
            
            # Unnormalize from [-1, 1] to [0, 1]
            input_tensor = input_tensor * 0.5 + 0.5
            output_tensor = output_tensor * 0.5 + 0.5
            
            # Output filenames
            clean_output_path = os.path.join(args.output_dir, f"{name}_output{ext}")
            comparison_path = os.path.join(args.output_dir, f"{name}_comparison{ext}")
            
            # Save translated image
            save_image(output_tensor.squeeze(0), clean_output_path)
            
            # Save side-by-side comparison
            if args.save_comparison:
                # Concatenate horizontally
                comparison_grid = torch.cat((input_tensor.squeeze(0), output_tensor.squeeze(0)), dim=2)
                save_image(comparison_grid, comparison_path)
                print(f"Saved: {clean_output_path} and {comparison_path}")
            else:
                print(f"Saved: {clean_output_path}")
    print("Inference completed.")
if __name__ == "__main__":
    main()
