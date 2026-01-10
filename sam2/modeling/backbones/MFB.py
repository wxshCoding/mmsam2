
import torch
import torch.nn as nn
import torch.nn.functional as F

class BasicConv2d(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_planes, out_planes,
                              kernel_size=kernel_size, stride=stride,
                              padding=padding, dilation=dilation, bias=False)
        self.bn = nn.BatchNorm2d(out_planes)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return x
class MFB_modified(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(MFB_modified, self).__init__()
        self.relu = nn.ReLU(True)
        self.branch0 = nn.Sequential(
            BasicConv2d(in_channel, out_channel, 1),
        )
        # self.branch0 = nn.Sequential(
        #     BasicConv2d(in_channel, out_channel, 1),
        # )
        self.branch1 = nn.Sequential(
            BasicConv2d(in_channel, out_channel, 1),
            BasicConv2d(out_channel, out_channel, kernel_size=(1, 3), padding=(0, 1)),
            BasicConv2d(out_channel, out_channel, kernel_size=(3, 1), padding=(1, 0)),
            BasicConv2d(out_channel, out_channel, 3, padding=3, dilation=3)
        )
        self.branch2 = nn.Sequential(
            BasicConv2d(in_channel, out_channel, 1),
            BasicConv2d(out_channel, out_channel, kernel_size=(1, 5), padding=(0, 2)),
            BasicConv2d(out_channel, out_channel, kernel_size=(5, 1), padding=(2, 0)),
            BasicConv2d(out_channel, out_channel, 3, padding=5, dilation=5)
        )
        self.branch3 = nn.Sequential(
            BasicConv2d(in_channel, out_channel, 1),
            BasicConv2d(out_channel, out_channel, kernel_size=(1, 7), padding=(0, 3)),
            BasicConv2d(out_channel, out_channel, kernel_size=(7, 1), padding=(3, 0)),
            BasicConv2d(out_channel, out_channel, 3, padding=7, dilation=7)
        )
        self.branch0_1_1 = nn.Sequential(
            BasicConv2d(in_channel, out_channel, 1),
            BasicConv2d(out_channel, out_channel, kernel_size=1)
        )
        self.branch0_3_1 = nn.Sequential(
            BasicConv2d(in_channel, out_channel, 1),
            BasicConv2d(out_channel, out_channel, kernel_size=3, padding=1)
        )
        self.branch0_5_1 = nn.Sequential(
            BasicConv2d(in_channel, out_channel, 1),
            BasicConv2d(out_channel, out_channel, kernel_size=5, padding=2)
        )
        self.branch0_7_1 = nn.Sequential(
            BasicConv2d(in_channel, out_channel, 1),
            BasicConv2d(out_channel, out_channel, kernel_size=7, padding=3)
        )
        # self.conv_cat = BasicConv2d(4*out_channel, out_channel, 3, padding=1)
        self.conv_cat = BasicConv2d(8*out_channel, out_channel, 3, padding=1)
        self.conv_res = BasicConv2d(in_channel, out_channel, 1)

    def forward(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        x3 = self.branch3(x)
        x1_1 = self.branch0_1_1(x)
        x3_1 = self.branch0_3_1(x)
        x5_1 = self.branch0_5_1(x)
        x7_1 = self.branch0_7_1(x)
        # x_cat = self.conv_cat(torch.cat((x0, x1, x2, x3), 1))
        x_cat = self.conv_cat(torch.cat((x0, x1, x2, x3,x1_1,x3_1,x5_1,x7_1), 1))
        # x_cat = self.conv_cat(torch.cat((x0, x1, x2, x3,x3_1,x5_1,x7_1), 1))

        x = self.relu(x_cat + self.conv_res(x))
        return x

def test_rfb_module():
    # 1. Create sample input
    batch_size = 2
    in_channels = 256  # 输入通道数
    out_channels = 128 # 输出通道数
    height = 64
    width = 64
    
    # 创建随机输入tensor
    x = torch.randn(batch_size, in_channels, height, width)
    
    # 2. Initialize RFB module
    mfb = MFB_modified(
        in_channel=in_channels,
        out_channel=out_channels
    )
    
    # 3. Forward pass
    output = mfb(x)
    
    # 4. Print shapes and verify
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    
    # 5. Verify branch outputs individually
    with torch.no_grad():
        # Test individual branches
        x0 = mfb.branch0(x)
        x1 = mfb.branch1(x)
        x2 = mfb.branch2(x)
        x3 = mfb.branch3(x)
        x3_1 = mfb.branch0_3_1(x)
        x5_1 = mfb.branch0_5_1(x)
        x7_1 = mfb.branch0_7_1(x)   

        
        print("\nBranch output shapes:")
        print(f"Branch 0: {x0.shape}")
        print(f"Branch 1: {x1.shape}")
        print(f"Branch 2: {x2.shape}")
        print(f"Branch 3: {x3.shape}")
        print(f"Branch 0_3_1: {x3_1.shape}")
        print(f"Branch 0_5_1: {x5_1.shape}")
        print(f"Branch 0_7_1: {x7_1.shape}")
        
        # Test concatenation
        x_cat = torch.cat((x0, x1, x2, x3,x3_1,x5_1,x7_1), 1)
        print(f"\nConcatenated shape: {x_cat.shape}")
        
        # Test residual connection
        res = mfb.conv_res(x)
        print(f"Residual shape: {res.shape}")    

if __name__ == "__main__":
    test_rfb_module()