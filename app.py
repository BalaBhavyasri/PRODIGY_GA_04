import os
import argparse
import base64
from io import BytesIO
from PIL import Image
import torch
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
from flask import Flask, request, jsonify, send_from_directory

from models import UNetGenerator

app = Flask(__name__, static_folder="static", static_url_path="")

# Global variables for model and device
generator = None
device = None
target_size = 256

def load_generator(checkpoint_path, dev):
    """Load model checkpoint into UNetGenerator."""
    net_g = UNetGenerator(input_nc=3, output_nc=3).to(dev)
    if os.path.exists(checkpoint_path):
        print(f"Loading checkpoint from {checkpoint_path}...")
        checkpoint = torch.load(checkpoint_path, map_location=dev)
        if 'generator_state_dict' in checkpoint:
            net_g.load_state_dict(checkpoint['generator_state_dict'])
        else:
            net_g.load_state_dict(checkpoint)
    else:
        print(f"Warning: Checkpoint {checkpoint_path} not found. Running with uninitialized weights.")
    net_g.eval()
    return net_g

def preprocess_image(pil_img):
    """Convert PIL image to preprocessed PyTorch tensor."""
    # Ensure RGB
    pil_img = pil_img.convert('RGB')
    # Resize and normalize
    transform = transforms.Compose([
        transforms.Resize((target_size, target_size), Image.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    return transform(pil_img).unsqueeze(0).to(device)

def postprocess_tensor(tensor):
    """Convert PyTorch tensor back to PIL Image."""
    # Denormalize
    tensor = tensor.squeeze(0).cpu() * 0.5 + 0.5
    # Clamp to [0, 1]
    tensor = torch.clamp(tensor, 0, 1)
    # Convert to PIL
    to_pil = transforms.ToPILImage()
    return to_pil(tensor)

@app.route('/')
def index():
    return send_from_directory('static', 'index.html')

@app.route('/translate', methods=['POST'])
def translate():
    try:
        data = request.get_json()
        if not data or 'image' not in data:
            return jsonify({'error': 'No image provided'}), 400
            
        # Parse base64 image data
        image_data = data['image'].split(',')[1] if ',' in data['image'] else data['image']
        image_bytes = base64.b64decode(image_data)
        input_image = Image.open(BytesIO(image_bytes))
        
        # Preprocess
        input_tensor = preprocess_image(input_image)
        
        # Inference
        with torch.no_grad():
            output_tensor = generator(input_tensor)
            
        # Postprocess
        output_image = postprocess_tensor(output_tensor)
        
        # Convert output to base64
        buffered = BytesIO()
        output_image.save(buffered, format="PNG")
        output_base64 = base64.b64encode(buffered.getvalue()).decode('utf-8')
        
        return jsonify({'image': f"data:image/png;base64,{output_base64}"})
        
    except Exception as e:
        print(f"Error during translation: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/presets', methods=['GET'])
def get_presets():
    """List available validation images for demonstration."""
    val_dir = os.path.join("facades", "val")
    if not os.path.exists(val_dir):
        return jsonify([])
        
    # Find all images
    files = sorted([
        f for f in os.listdir(val_dir)
        if f.lower().endswith(('.png', '.jpg', '.jpeg'))
    ])
    return jsonify(files[:20]) # limit to first 20 for loading performance

@app.route('/presets/<filename>', methods=['GET'])
def get_preset_image(filename):
    """Serve a split or full preset image from validation set."""
    val_dir = os.path.join("facades", "val")
    img_path = os.path.join(val_dir, filename)
    if not os.path.exists(img_path):
        return jsonify({'error': 'Image not found'}), 404
        
    # Standard combined image: Left is target, Right is input (label map)
    combined = Image.open(img_path)
    w, h = combined.size
    
    # Crop into two parts
    # If the combined image width is double the height, split it.
    if w == h * 2:
        img_a = combined.crop((0, 0, w // 2, h)) # Label Map / Input B
        img_b = combined.crop((w // 2, 0, w, h)) # Real Photo / Target A
    else:
        # If not double-width, just return the whole image as both
        img_a = combined
        img_b = combined
        
    # Convert both to base64
    buffered_a = BytesIO()
    img_a.save(buffered_a, format="JPEG")
    base64_a = base64.b64encode(buffered_a.getvalue()).decode('utf-8')
    
    buffered_b = BytesIO()
    img_b.save(buffered_b, format="JPEG")
    base64_b = base64.b64encode(buffered_b.getvalue()).decode('utf-8')
    
    return jsonify({
        'label': f"data:image/jpeg;base64,{base64_a}",
        'photo': f"data:image/jpeg;base64,{base64_b}"
    })

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Run the Pix2Pix Interactive Web Server")
    parser.add_argument("--checkpoint", type=str, default="./checkpoints/pix2pix_epoch_005.pth", help="Path to trained generator model checkpoint")
    parser.add_argument("--port", type=int, default=5000, help="Port to run the Flask server on")
    parser.add_argument("--host", type=str, default="127.0.0.1", help="Host ip to bind")
    
    args = parser.parse_args()
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load model
    generator = load_generator(args.checkpoint, device)
    
    # Start app
    app.run(host=args.host, port=args.port, debug=False)
