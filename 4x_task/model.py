##################
# 4x-task
##################

import torch
import torch.nn as nn
import torch.nn.functional as F

###############
# model
class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, use_bn=True):
        super(ConvBlock, self).__init__()
        self.use_bn = use_bn
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        self.bn = nn.BatchNorm2d(out_channels) if use_bn else None
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.conv(x)
        if self.use_bn:
            x = self.bn(x)
        x = self.relu(x)
        return x

# channel attention
class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc1   = nn.Conv2d(in_planes, in_planes // 16, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2   = nn.Conv2d(in_planes // 16, in_planes, 1, bias=False)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return self.sigmoid(out)  

# residual block
class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = ConvBlock(channels, channels, kernel_size=3, stride=1, padding=1)
        self.conv2 = ConvBlock(channels, channels, kernel_size=3, stride=1, padding=1, use_bn=False)
        self.ca = ChannelAttention(channels)

    def forward(self, x):
        identity = x
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.ca(x) * x
        return x + identity

# Dense block
class DenseBlock(nn.Module):
    def __init__(self, in_channels, growth_rate, num_layers):
        super(DenseBlock, self).__init__()
        self.layers = nn.ModuleList([
            ConvBlock(in_channels + i*growth_rate, growth_rate, kernel_size=3, stride=1, padding=1)
            for i in range(num_layers)
        ])

    def forward(self, x):
        for layer in self.layers:
            out = layer(x)
            x = torch.cat([x, out], dim=1)
        return x

# self-attention
class SelfAttentionBlock(nn.Module):
    def __init__(self, dim, num_heads):
        super(SelfAttentionBlock, self).__init__()
        self.attention = nn.MultiheadAttention(dim, num_heads)

    def forward(self, x):
        # x shape: (batch_size, num_patches, embedding_dim)
        x = x.permute(1, 0, 2)  # shape: (num_patches, batch_size, embedding_dim)
        attn_output, _ = self.attention(x, x, x)
        return attn_output.permute(1, 0, 2)  # shape: (batch_size, num_patches, embedding_dim)

# upsample block
class UpSampleBlock(nn.Module):
    def __init__(self, in_channels, out_channels, scale_factor):
        super(UpSampleBlock, self).__init__()
        self.conv = ConvBlock(in_channels, in_channels * (scale_factor ** 2), kernel_size=3, stride=1, padding=1)
        self.pixel_shuffle = nn.PixelShuffle(scale_factor)

    def forward(self, x):
        x = self.conv(x)
        x = self.pixel_shuffle(x)
        return x

# 1. Correction block
class CorrectionBlock(nn.Module):
    def __init__(self, in_channels, num_res_blocks, num_heads, patch_dim, growth_rate, num_dense_layers):
        super(CorrectionBlock, self).__init__()
        self.patch_dim = patch_dim
        self.patch_to_embedding = nn.Linear(patch_dim * patch_dim, patch_dim * patch_dim)
        self.self_attention = SelfAttentionBlock(patch_dim*patch_dim, num_heads)
        self.block1 = nn.Sequential(
            DenseBlock((in_channels+1)*2, growth_rate, num_dense_layers),
            ConvBlock((in_channels+1)*2 + growth_rate * num_dense_layers, 128, kernel_size=3, stride=1, padding=1),
            *[ResidualBlock(128) for _ in range(num_res_blocks)],
            ConvBlock(128, 64, kernel_size=3, stride=1, padding=1),
            ConvBlock(64, in_channels, kernel_size=3, stride=1, padding=1, use_bn=False)
        )
        
    def forward(self, LR_prediction, LR_geo):
        
        # Concatenate the input data
        LR_input = torch.cat([LR_prediction, LR_geo], dim=1)
        # Transform the input data into patches
        LR_input_patches = LR_input.unfold(2, self.patch_dim, self.patch_dim).unfold(3, self.patch_dim, self.patch_dim)
        LR_input_patches = LR_input_patches.contiguous().view(LR_input.size(0), -1, self.patch_dim * self.patch_dim)

        # Apply self-attention
        LR_input_patches = self.self_attention(self.patch_to_embedding(LR_input_patches))
        # Reshape the data back to the original shape
        LR_input1 = LR_input_patches.contiguous().view(LR_input.size(0), -1, LR_input.size(1))
        LR_input1 = LR_input1.permute(0, 2, 1)
        LR_input1 = LR_input1.contiguous().view(LR_input.size(0), LR_input.size(1), LR_input.size(2), LR_input.size(3))  
        # Apply the rest of the model
        LR_input2 = torch.cat([LR_input, LR_input1], dim=1)
        LR_real = self.block1(LR_input2)
        return LR_real

# 2. Downscaling block    
class DownscalingBlock(nn.Module):
    def __init__(self, in_channels, num_res_blocks, num_dense_layers, growth_rate):
        super(DownscalingBlock, self).__init__()
        self.block1 = DenseBlock(in_channels+1, growth_rate, num_dense_layers)
        self.block2 = nn.Sequential(
            ConvBlock(in_channels+1 + num_dense_layers*growth_rate, 128, kernel_size=3, stride=1, padding=1),
            *[ResidualBlock(128) for _ in range(num_res_blocks)],
            UpSampleBlock(128,128,scale_factor=4),
            ConvBlock(128, 64, kernel_size=3, stride=1, padding=1),
            ConvBlock(64, in_channels//2, kernel_size=3, stride=1, padding=1, use_bn=False)
        )

    def forward(self, LR_prediction, corrected_LR_prediction, LR_geo):
        
        LR_input = torch.cat([LR_prediction, corrected_LR_prediction, LR_geo], dim=1)
        LR_input = self.block1(LR_input)
        HR_real = self.block2(LR_input)
        return HR_real

# 3. DSAF  
class DSAF(nn.Module):
    def __init__(self, in_channels, num_res_blocks, num_heads, patch_dim, growth_rate, num_dense_layers, in_channels1, num_res_blocks1, num_dense_layers1, growth_rate1, init_weights):
        super(DSAF, self).__init__()

        self.correction_block = CorrectionBlock(in_channels, num_res_blocks, num_heads, patch_dim, growth_rate, num_dense_layers)
        self.downscaling_block = DownscalingBlock(in_channels1, num_res_blocks1, num_dense_layers1, growth_rate1)
        self.weights = nn.Parameter(init_weights)

    def forward(self, LR_prediction, LR_geo):
        corrected_LR_prediction = self.correction_block(LR_prediction, LR_geo)
        HR_real = self.downscaling_block(LR_prediction, corrected_LR_prediction, LR_geo)
        return corrected_LR_prediction, HR_real, self.weights