import os
import time
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision.utils import save_image
from dataset import Pix2PixDataset
from models import UNetGenerator, PatchGANDiscriminator, init_weights
def main():
    parser = argparse.ArgumentParser(description="Train a Pix2Pix conditional GAN (cGAN)")
    parser.add_argument("--dataroot", type=str, required=True, help="Path to combined dataset directory (must contain 'train' and 'val' subfolders)")
    parser.add_argument("--direction", type=str, default="a2b", choices=["a2b", "b2a"], help="Direction of translation (a2b: left to right, b2a: right to left)")
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size (Pix2Pix paper defaults to 1)")
    parser.add_argument("--lr", type=float, default=0.0002, help="Learning rate for Adam")
    parser.add_argument("--beta1", type=float, default=0.5, help="Beta1 hyperparameter for Adam optimizer")
    parser.add_argument("--beta2", type=float, default=0.999, help="Beta2 hyperparameter for Adam optimizer")
    parser.add_argument("--lambda_L1", type=float, default=100.0, help="Weight for L1 loss")
    parser.add_argument("--epochs", type=int, default=200, help="Total number of training epochs")
    parser.add_argument("--epoch_decay_start", type=int, default=100, help="Epoch to start linear learning rate decay to 0")
    parser.add_argument("--save_epoch_freq", type=int, default=20, help="Frequency of saving checkpoints (epochs)")
    parser.add_argument("--num_workers", type=int, default=4, help="Number of worker threads for DataLoader")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Device to use ('cuda' or 'cpu')")
    parser.add_argument("--output_dir", type=str, default="outputs", help="Directory to save sample validation outputs")
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints", help="Directory to save model checkpoints")
    
    args = parser.parse_args()
    
    # Create directories
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    
    print(f"Using device: {args.device}")
    
    # Paths for train and val splits
    train_dir = os.path.join(args.dataroot, "train")
    val_dir = os.path.join(args.dataroot, "val")
    
    if not os.path.exists(train_dir):
        raise ValueError(f"Train directory not found at: {train_dir}")
    if not os.path.exists(val_dir):
        print(f"Warning: Validation directory not found at: {val_dir}. Validation stages will be skipped.")
        has_val = False
    else:
        has_val = True
    # 1. Datasets & Dataloaders
    train_dataset = Pix2PixDataset(train_dir, direction=args.direction, is_train=True)
    train_loader = DataLoader(
        train_dataset, 
        batch_size=args.batch_size, 
        shuffle=True, 
        num_workers=args.num_workers,
        pin_memory=True if args.device == "cuda" else False
    )
    print(f"Train dataset loaded: {len(train_dataset)} images.")
    
    if has_val:
        val_dataset = Pix2PixDataset(val_dir, direction=args.direction, is_train=False)
        val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False, num_workers=args.num_workers)
        print(f"Validation dataset loaded: {len(val_dataset)} images.")
    # 2. Instantiate Networks & Init Weights
    generator = UNetGenerator(input_nc=3, output_nc=3).to(args.device)
    discriminator = PatchGANDiscriminator(input_nc=6).to(args.device) # input_nc=6 (input + target concatenated)
    
    init_weights(generator)
    init_weights(discriminator)
    # 3. Optimizers
    optimizer_G = optim.Adam(generator.parameters(), lr=args.lr, betas=(args.beta1, args.beta2))
    optimizer_D = optim.Adam(discriminator.parameters(), lr=args.lr, betas=(args.beta1, args.beta2))
    # 4. Schedulers (Linear decay)
    def lr_lambda(epoch):
        if epoch < args.epoch_decay_start:
            return 1.0
        return 1.0 - float(epoch - args.epoch_decay_start) / float(args.epochs - args.epoch_decay_start)
        
    scheduler_G = optim.lr_scheduler.LambdaLR(optimizer_G, lr_lambda=lr_lambda)
    scheduler_D = optim.lr_scheduler.LambdaLR(optimizer_D, lr_lambda=lr_lambda)
    # 5. Loss Functions
    criterion_GAN = nn.BCEWithLogitsLoss()
    criterion_L1 = nn.L1Loss()
    # Helper function to save sample visual output
    def save_sample_results(epoch, loader, num_samples=4):
        generator.eval()
        with torch.no_grad():
            inputs, targets = next(iter(loader))
            inputs, targets = inputs.to(args.device), targets.to(args.device)
            fakes = generator(inputs)
            
            # Unnormalize from [-1, 1] to [0, 1] for saving
            inputs = inputs * 0.5 + 0.5
            targets = targets * 0.5 + 0.5
            fakes = fakes * 0.5 + 0.5
            
            # Combine input, fake, target side-by-side
            results = []
            for k in range(min(inputs.size(0), num_samples)):
                grid = torch.cat([inputs[k], fakes[k], targets[k]], dim=2) # concatenate horizontally
                results.append(grid)
            
            results_grid = torch.cat(results, dim=1) # concatenate vertically
            save_path = os.path.join(args.output_dir, f"epoch_{epoch:03d}.png")
            save_image(results_grid, save_path)
            print(f"Saved epoch samples to {save_path}")
        generator.train()
    # 6. Training Loop
    total_steps = 0
    start_time = time.time()
    
    for epoch in range(1, args.epochs + 1):
        epoch_start_time = time.time()
        print(f"\n--- Epoch {epoch}/{args.epochs} (LR: {optimizer_G.param_groups[0]['lr']:.6f}) ---")
        
        generator.train()
        discriminator.train()
        
        for i, (real_A, real_B) in enumerate(train_loader):
            real_A = real_A.to(args.device)
            real_B = real_B.to(args.device)
            
            # ---------------------
            #  Train Discriminator
            # ---------------------
            optimizer_D.zero_grad()
            
            # Real pair (Input A, Target B)
            real_pair = torch.cat((real_A, real_B), 1)
            pred_real = discriminator(real_pair)
            loss_D_real = criterion_GAN(pred_real, torch.ones_like(pred_real))
            
            # Fake pair (Input A, Generated B)
            fake_B = generator(real_A)
            fake_pair = torch.cat((real_A, fake_B.detach()), 1)
            pred_fake = discriminator(fake_pair)
            loss_D_fake = criterion_GAN(pred_fake, torch.zeros_like(pred_fake))
            
            # Total Discriminator Loss
            loss_D = (loss_D_real + loss_D_fake) * 0.5
            loss_D.backward()
            optimizer_D.step()
            
            # -----------------
            #  Train Generator
            # -----------------
            optimizer_G.zero_grad()
            
            # Adversarial Loss (Discriminator should predict Fake pair as Real)
            fake_pair_G = torch.cat((real_A, fake_B), 1)
            pred_fake_G = discriminator(fake_pair_G)
            loss_G_GAN = criterion_GAN(pred_fake_G, torch.ones_like(pred_fake_G))
            
            # Content Loss (L1 loss between Generated B and Target B)
            loss_G_L1 = criterion_L1(fake_B, real_B)
            
            # Total Generator Loss
            loss_G = loss_G_GAN + args.lambda_L1 * loss_G_L1
            loss_G.backward()
            optimizer_G.step()
            
            total_steps += 1
            
            # Log progress
            if i % 100 == 0 or i == len(train_loader) - 1:
                elapsed = time.time() - start_time
                print(f"[Epoch {epoch:03d}/{args.epochs:03d}] [Batch {i:03d}/{len(train_loader):03d}] "
                      f"Loss D: {loss_D.item():.4f} | Loss G (GAN): {loss_G_GAN.item():.4f} | "
                      f"Loss G (L1): {loss_G_L1.item():.4f} | Time: {elapsed:.1f}s")
                      
        # Update learning rates
        scheduler_G.step()
        scheduler_D.step()
        
        # Save sample outputs
        if has_val and (epoch == 1 or epoch % 5 == 0 or epoch == args.epochs):
            save_sample_results(epoch, val_loader)
            
        # Save Checkpoint
        if epoch % args.save_epoch_freq == 0 or epoch == args.epochs:
            checkpoint_path = os.path.join(args.checkpoint_dir, f"pix2pix_epoch_{epoch:03d}.pth")
            torch.save({
                'epoch': epoch,
                'generator_state_dict': generator.state_dict(),
                'discriminator_state_dict': discriminator.state_dict(),
                'optimizer_G_state_dict': optimizer_G.state_dict(),
                'optimizer_D_state_dict': optimizer_D.state_dict(),
                'loss_G': loss_G.item(),
                'loss_D': loss_D.item(),
            }, checkpoint_path)
            print(f"Saved model checkpoint to {checkpoint_path}")
            
        print(f"Epoch {epoch} finished in {time.time() - epoch_start_time:.1f}s")
        
    print(f"\nTraining completed in {time.time() - start_time:.1f}s.")
if __name__ == "__main__":
    main()