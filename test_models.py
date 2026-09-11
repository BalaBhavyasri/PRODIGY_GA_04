import torch
from models import UNetGenerator, PatchGANDiscriminator, init_weights
def test_unet_generator():
    print("\n--- Testing U-Net Generator ---")
    batch_size = 2
    input_channels = 3
    output_channels = 3
    height, width = 256, 256
    
    # Instantiate U-Net Generator
    net_g = UNetGenerator(input_nc=input_channels, output_nc=output_channels)
    print("UNetGenerator instantiated successfully.")
    
    # Test initialization
    init_weights(net_g)
    print("UNetGenerator weights initialized successfully.")
    
    # Create random input tensor
    x = torch.randn(batch_size, input_channels, height, width)
    print(f"Input shape: {x.shape}")
    
    # Forward pass
    with torch.no_grad():
        out = net_g(x)
        
    print(f"Output shape: {out.shape}")
    
    # Verify shape
    expected_shape = (batch_size, output_channels, height, width)
    assert out.shape == expected_shape, f"Expected output shape {expected_shape}, but got {out.shape}"
    print("U-Net Generator output shape verification PASSED!")
def test_patchgan_discriminator():
    print("\n--- Testing PatchGAN Discriminator ---")
    batch_size = 2
    input_channels = 6 # Concatenation of input (3 ch) and target (3 ch)
    height, width = 256, 256
    
    # Instantiate 70x70 PatchGAN Discriminator (n_layers=3)
    net_d = PatchGANDiscriminator(input_nc=input_channels)
    print("PatchGANDiscriminator instantiated successfully.")
    
    # Test initialization
    init_weights(net_d)
    print("PatchGANDiscriminator weights initialized successfully.")
    
    # Create random input tensor
    x = torch.randn(batch_size, input_channels, height, width)
    print(f"Input shape: {x.shape}")
    
    # Forward pass
    with torch.no_grad():
        out = net_d(x)
        
    print(f"Discriminator Output shape: {out.shape}")
    
    # Verify shape
    # For a 256x256 image and 3 convolutional layers with stride 2 and then two layers with stride 1,
    # the output size should be batch_size x 1 x 30 x 30.
    expected_shape = (batch_size, 1, 30, 30)
    assert out.shape == expected_shape, f"Expected output shape {expected_shape}, but got {out.shape}"
    print("PatchGAN Discriminator output shape verification PASSED!")
if __name__ == "__main__":
    print("Starting Pix2Pix Model verification tests...")
    try:
        test_unet_generator()
        test_patchgan_discriminator()
        print("\nAll model tests PASSED successfully!")
    except Exception as e:
        print(f"\nTest FAILED with exception: {e}")
        import traceback
        traceback.print_exc()