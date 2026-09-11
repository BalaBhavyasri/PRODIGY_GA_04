import torch
import torch.nn as nn
def init_weights(net, init_type='normal', init_gain=0.02):
    """
    Initialize network weights.
    Referenced from the official pix2pix implementation.
    """
    def init_func(m):
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            if init_type == 'normal':
                nn.init.normal_(m.weight.data, 0.0, init_gain)
            elif init_type == 'xavier':
                nn.init.xavier_normal_(m.weight.data, gain=init_gain)
            elif init_type == 'kaiming':
                nn.init.kaiming_normal_(m.weight.data, a=0.2, mode='fan_in')
            elif init_type == 'orthogonal':
                nn.init.orthogonal_(m.weight.data, gain=init_gain)
            else:
                raise NotImplementedError(f'initialization method [{init_type}] is not implemented')
            if hasattr(m, 'bias') and m.bias is not None:
                nn.init.constant_(m.bias.data, 0.0)
        elif classname.find('BatchNorm2d') != -1:
            nn.init.normal_(m.weight.data, 1.0, init_gain)
            nn.init.constant_(m.bias.data, 0.0)
    print(f'Initializing weights using [{init_type}]')
    net.apply(init_func)
    return net
class UNetSkipConnectionBlock(nn.Module):
    """
    Defines the U-Net submodule with skip connection.
    X -------------------> (concat) -------------------> output
      |-- downsampling -- submodule -- upsampling --|
    """
    def __init__(self, outer_nc, inner_nc, input_nc=None,
                 submodule=None, outermost=False, innermost=False, norm_layer=nn.BatchNorm2d, use_dropout=False):
        super(UNetSkipConnectionBlock, self).__init__()
        self.outermost = outermost
        
        # Default input channels to outer channels
        if input_nc is None:
            input_nc = outer_nc
            
        downconv = nn.Conv2d(input_nc, inner_nc, kernel_size=4, stride=2, padding=1, bias=False)
        downrelu = nn.LeakyReLU(0.2, True)
        downnorm = norm_layer(inner_nc)
        uprelu = nn.ReLU(True)
        upnorm = norm_layer(outer_nc)
        if outermost:
            upconv = nn.ConvTranspose2d(inner_nc * 2, outer_nc, kernel_size=4, stride=2, padding=1)
            down = [downconv]
            up = [uprelu, upconv, nn.Tanh()]
            model = down + [submodule] + up
        elif innermost:
            upconv = nn.ConvTranspose2d(inner_nc, outer_nc, kernel_size=4, stride=2, padding=1, bias=False)
            down = [downrelu, downconv]
            up = [uprelu, upconv, upnorm]
            model = down + up
        else:
            upconv = nn.ConvTranspose2d(inner_nc * 2, outer_nc, kernel_size=4, stride=2, padding=1, bias=False)
            down = [downrelu, downconv, downnorm]
            up = [uprelu, upconv, upnorm]
            if use_dropout:
                up += [nn.Dropout(0.5)]
            model = down + [submodule] + up
        self.model = nn.Sequential(*model)
    def forward(self, x):
        if self.outermost:
            return self.model(x)
        else:
            # Concatenate along the channel dimension
            return torch.cat([x, self.model(x)], 1)
class UNetGenerator(nn.Module):
    """
    U-Net based generator.
    For 256x256 input/output images.
    """
    def __init__(self, input_nc=3, output_nc=3, num_filters=64, norm_layer=nn.BatchNorm2d, use_dropout=True):
        super(UNetGenerator, self).__init__()
        
        # Build U-Net structure recursively from the innermost block to the outermost block.
        # Innermost block (resolution 1x1)
        unet_block = UNetSkipConnectionBlock(
            outer_nc=num_filters * 8, inner_nc=num_filters * 8, input_nc=None,
            submodule=None, innermost=True, norm_layer=norm_layer
        )
        
        # Add intermediate blocks with dropout (resolutions 2x2, 4x4, 8x8)
        unet_block = UNetSkipConnectionBlock(
            outer_nc=num_filters * 8, inner_nc=num_filters * 8, input_nc=None,
            submodule=unet_block, norm_layer=norm_layer, use_dropout=use_dropout
        )
        unet_block = UNetSkipConnectionBlock(
            outer_nc=num_filters * 8, inner_nc=num_filters * 8, input_nc=None,
            submodule=unet_block, norm_layer=norm_layer, use_dropout=use_dropout
        )
        unet_block = UNetSkipConnectionBlock(
            outer_nc=num_filters * 8, inner_nc=num_filters * 8, input_nc=None,
            submodule=unet_block, norm_layer=norm_layer, use_dropout=use_dropout
        )
        
        # Add intermediate blocks without dropout (resolutions 16x16, 32x32, 64x64, 128x128)
        unet_block = UNetSkipConnectionBlock(
            outer_nc=num_filters * 4, inner_nc=num_filters * 8, input_nc=None,
            submodule=unet_block, norm_layer=norm_layer
        )
        unet_block = UNetSkipConnectionBlock(
            outer_nc=num_filters * 2, inner_nc=num_filters * 4, input_nc=None,
            submodule=unet_block, norm_layer=norm_layer
        )
        unet_block = UNetSkipConnectionBlock(
            outer_nc=num_filters, inner_nc=num_filters * 2, input_nc=None,
            submodule=unet_block, norm_layer=norm_layer
        )
        
        # Outermost block (resolution 256x256)
        self.model = UNetSkipConnectionBlock(
            outer_nc=output_nc, inner_nc=num_filters, input_nc=input_nc,
            submodule=unet_block, outermost=True, norm_layer=norm_layer
        )
    def forward(self, x):
        return self.model(x)
class PatchGANDiscriminator(nn.Module):
    """
    Defines a PatchGAN discriminator.
    With n_layers=3, this forms a 70x70 PatchGAN discriminator.
    """
    def __init__(self, input_nc=6, ndf=64, n_layers=3, norm_layer=nn.BatchNorm2d):
        super(PatchGANDiscriminator, self).__init__()
        
        # Use bias=False for conv layers since we use batch/instance normalization
        use_bias = norm_layer == nn.InstanceNorm2d
        
        model = [
            nn.Conv2d(input_nc, ndf, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, True)
        ]
        
        # Incrementally double filters up to 8 * ndf
        nf_mult = 1
        nf_mult_prev = 1
        for n in range(1, n_layers):
            nf_mult_prev = nf_mult
            nf_mult = min(2 ** n, 8)
            model += [
                nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=4, stride=2, padding=1, bias=use_bias),
                norm_layer(ndf * nf_mult),
                nn.LeakyReLU(0.2, True)
            ]
            
        # Add penultimate layer with stride=1
        nf_mult_prev = nf_mult
        nf_mult = min(2 ** n_layers, 8)
        model += [
            nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=4, stride=1, padding=1, bias=use_bias),
            norm_layer(ndf * nf_mult),
            nn.LeakyReLU(0.2, True)
        ]
        
        # Final layer outputs a single-channel spatial prediction map
        model += [
            nn.Conv2d(ndf * nf_mult, 1, kernel_size=4, stride=1, padding=1)
        ]
        
        self.model = nn.Sequential(*model)
    def forward(self, x):
        return self.model(x)
