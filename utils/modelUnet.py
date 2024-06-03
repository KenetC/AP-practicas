import torch
import torch.nn as nn

class Conv_3_k(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
    def forward(self, x):
        return self.conv1(x)
class Double_Conv(nn.Module):
    '''
    Double convolution block for U-Net
    '''
    def __init__(self, in_channels, out_channels,BN:bool):
        super().__init__()
        
        if BN: 
            self.double_conv = nn.Sequential(
                Conv_3_k(in_channels, out_channels),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(),
                Conv_3_k(out_channels, out_channels),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(),
                )
        else: 
            self.double_conv = nn.Sequential(
                Conv_3_k(in_channels, out_channels),
                nn.ReLU(),
                Conv_3_k(out_channels, out_channels),
                nn.ReLU(),
                )

    def forward(self, x):
        return self.double_conv(x)
    


class Down_Conv(nn.Module):
    '''
    Down convolution part
    '''
    def __init__(self, in_channels, out_channels,BN:bool):
        super().__init__()
        self.encoder = nn.Sequential(
                        nn.MaxPool2d(2,2),
                        Double_Conv(in_channels, out_channels,BN)
                        )
    def forward(self, x):
        return self.encoder(x)
    
class Up_Conv(nn.Module):
    '''
    Up convolution part
    '''
    def __init__(self,in_channels, out_channels,BN:bool):
        super().__init__()
        self.upsample_layer = nn.Sequential(
                        nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
                        nn.Conv2d(in_channels,in_channels//2,kernel_size=1,stride=1)
                        )
        self.decoder = Double_Conv(in_channels, out_channels,BN)
    
    def forward(self, x1, x2):
        '''
        x1 - upsampled volume
        x2 - volume from down sample to concatenate
        '''
        x1 = self.upsample_layer(x1)
        # Problema con el tema de la concatenacion no me termina de cerrar porque pasa esto!!! 
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = nn.functional.pad(x1, [diffX // 2, diffX - diffX // 2,
                                    diffY // 2, diffY - diffY // 2])
        x = torch.cat([x2,x1],dim=1)
        return self.decoder(x)
class UNet(nn.Module):
    def __init__(self, n_channels,BN:bool,base:int=32):
        super().__init__()
        #base = 32 #Número base de feature maps que luego irán duplicando.

        self.first_conv = Double_Conv(n_channels,base,BN)
        self.conv_down1 = Down_Conv(base , 2*base,BN)
        self.conv_down2 = Down_Conv(2*base , 4*base,BN)
        self.conv_down3 = Down_Conv(4*base , 8*base,BN)
        
        self.middle_conv = Double_Conv(8*base , 16*base,BN)

        self.conv_up1 = Up_Conv(16*base, 8*base,BN)
        self.conv_up2 = Up_Conv(8*base, 4*base,BN)
        self.conv_up3 = Up_Conv(4*base, 2*base,BN)
        self.conv_up4 = Up_Conv(2*base, base,BN)

        self.conv_last = nn.Conv2d(base,n_channels,kernel_size=1,stride=1)
        self.sigmoid = nn.Sigmoid()         #Sigmoide para usar al final
    def forward(self, x):
        x1 = self.first_conv(x)
        x2 = self.conv_down1(x1)
        x3 = self.conv_down2(x2)
        x4 = self.conv_down3(x3)
        
        x5 = self.middle_conv(x4)
        
        u1 = self.conv_up1(x5, x4)
        u2 = self.conv_up2(u1, x3)
        u3 = self.conv_up3(u2, x2)
        u4 = self.conv_up4(u3, x1)

        x = self.conv_last(u4)

        out = self.sigmoid(x)
        return out