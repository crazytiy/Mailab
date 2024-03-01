import numpy as np
import torch
from torch import nn
from torch import optim
import torch.nn.functional as F


class SEUnet(nn.Module):
    """
    """
    def __init__(self, configs):
        super(SEUnet, self).__init__()

        self.configs = configs
        self.input_dim = configs.in_channels * configs.input_length
        self.num_filter = configs.num_filter
        self.output_dim = configs.out_channels * configs.target_length
        self.Res = configs.res_layer
        self.ups = configs.upsample
        self.batch_norm = configs.batch_norm
        self.se = configs.senet

        # Encoder
        self.conv1_1 = ConvBlock(configs.input_length,
                                 self.num_filter,
                                 activation=False,
                                 BasicBlock=self.Res,
                                 batch_norm=self.batch_norm,
                                 se=self.se)
        self.conv2_1 = ConvBlock(self.num_filter,
                                 self.num_filter * 2,
                                 BasicBlock=self.Res,
                                 batch_norm=self.batch_norm,
                                 se=self.se)
        self.conv3_1 = ConvBlock(self.num_filter * 2,
                                 self.num_filter * 4,
                                 BasicBlock=self.Res,
                                 batch_norm=self.batch_norm,
                                 se=self.se)
        self.conv4_1 = ConvBlock(self.num_filter * 4,
                                 self.num_filter * 8,
                                 BasicBlock=self.Res,
                                 batch_norm=self.batch_norm,
                                 se=self.se)
        self.conv5_1 = ConvBlock(self.num_filter * 8,
                                 self.num_filter * 8,
                                 BasicBlock=self.Res,
                                 batch_norm=self.batch_norm,
                                 se=self.se)
        if 0:
            self.conv6_1 = ConvBlock(self.num_filter * 8,
                                     self.num_filter * 8,
                                     BasicBlock=self.Res,
                                     batch_norm=self.batch_norm,
                                     se=self.se)
            self.conv7_1 = ConvBlock(self.num_filter * 8,
                                     self.num_filter * 8,
                                     BasicBlock=self.Res,
                                     batch_norm=self.batch_norm,
                                     se=self.se)

        # Encoder
        self.conv1_2 = ConvBlock(configs.input_length,
                                 self.num_filter,
                                 activation=False,
                                 BasicBlock=self.Res,
                                 batch_norm=self.batch_norm,
                                 se=self.se)
        self.conv2_2 = ConvBlock(self.num_filter,
                                 self.num_filter * 2,
                                 BasicBlock=self.Res,
                                 batch_norm=self.batch_norm,
                                 se=self.se)
        self.conv3_2 = ConvBlock(self.num_filter * 2,
                                 self.num_filter * 4,
                                 BasicBlock=self.Res,
                                 batch_norm=self.batch_norm,
                                 se=self.se)
        self.conv4_2 = ConvBlock(self.num_filter * 4,
                                 self.num_filter * 8,
                                 BasicBlock=self.Res,
                                 batch_norm=self.batch_norm,
                                 se=self.se)
        self.conv5_2 = ConvBlock(self.num_filter * 8,
                                 self.num_filter * 8,
                                 BasicBlock=self.Res,
                                 batch_norm=self.batch_norm,
                                 se=self.se)
        if 0:
            self.conv6_2 = ConvBlock(self.num_filter * 8,
                                     self.num_filter * 8,
                                     BasicBlock=self.Res,
                                     batch_norm=self.batch_norm,
                                     se=self.se)
            self.conv7_2 = ConvBlock(self.num_filter * 8,
                                     self.num_filter * 8,
                                     BasicBlock=self.Res,
                                     batch_norm=self.batch_norm,
                                     se=self.se)

        # Decoder
        self.deconv1 = DeconvBlock(self.num_filter * 8 * 2,
                                   self.num_filter * 8 * 2,
                                   dropout=True,
                                   ups=self.ups,
                                   batch_norm=self.batch_norm,
                                   se=self.se)
        self.deconv2 = DeconvBlock(self.num_filter * 8 * 2 * 2,
                                   self.num_filter * 8,
                                   ups=self.ups,
                                   batch_norm=self.batch_norm,
                                   se=self.se)
        self.deconv3 = DeconvBlock(self.num_filter * 8 * 2,
                                   self.num_filter * 4,
                                   ups=self.ups,
                                   batch_norm=self.batch_norm,
                                   se=self.se)
        self.deconv4 = DeconvBlock(self.num_filter * 4 * 2,
                                   self.num_filter * 2,
                                   ups=self.ups,
                                   batch_norm=self.batch_norm,
                                   se=self.se)
        self.deconv5 = DeconvBlock(self.num_filter * 2 * 2,
                                   #self.output_dim,
                                   self.num_filter * 2,
                                   batch_norm=False,
                                   ups=self.ups,
                                   se=self.se)

        self.outconv = OutConv(self.num_filter*2, self.output_dim)


    def forward(self, x):
        bs, frames, channels, height, width = x.shape
        inp1, inp2 = x[:, :, 0], x[:, :, 1]

        inp1 = inp1.reshape(bs, -1, height, width)
        inp2 = inp2.reshape(bs, -1, height, width)
        # Encoder
        enc1_1 = self.conv1_1(inp1)
        enc2_1 = self.conv2_1(enc1_1)
        enc3_1 = self.conv3_1(enc2_1)
        enc4_1 = self.conv4_1(enc3_1)
        enc5_1 = self.conv5_1(enc4_1)
        #enc6_1 = self.conv6(enc5_1)
        #enc7_1 = self.conv7(enc6_1)

        enc1_2 = self.conv1_2(inp2)
        enc2_2 = self.conv2_2(enc1_2)
        enc3_2 = self.conv3_2(enc2_2)
        enc4_2 = self.conv4_2(enc3_2)
        enc5_2 = self.conv5_2(enc4_2)
        #enc6_2 = self.conv6(enc5_2)
        #enc7_2 = self.conv7(enc6_2)

        # Decoder with skip-connections
        dec1 = self.deconv1(torch.cat([enc5_1, enc5_2], dim=1))
        dec1 = _concat(dec1, torch.cat([enc4_1, enc4_2], dim=1))
        dec2 = self.deconv2(dec1)
        dec2 = _concat(dec2, torch.cat([enc3_1, enc3_2], dim=1))
        dec3 = self.deconv3(dec2)
        dec3 = _concat(dec3, torch.cat([enc2_1, enc2_2], dim=1))
        dec4 = self.deconv4(dec3)
        dec4 = _concat(dec4, torch.cat([enc1_1, enc1_2], dim=1))
        dec5 = self.deconv5(dec4)

        #out = torch.nn.ReLU6()(dec5) / 6
        out = self.outconv(dec5)
        out = out.reshape(bs, self.configs.target_length, self.configs.out_channels, height, width)

        return out


    def normal_weight_init(self, mean=0.0, std=0.02):
        for m in self.children():
            if isinstance(m, ConvBlock):
                nn.init.normal(m.conv.weight, mean, std)
            if isinstance(m, DeconvBlock):
                nn.init.normal(m.deconv.weight, mean, std)


def _concat(x1, x2):
    diffY = x2.size()[2] - x1.size()[2]
    diffX = x2.size()[3] - x1.size()[3]

    x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                    diffY // 2, diffY - diffY // 2])

    return torch.cat([x1, x2], dim=1)


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)


class ResBlock(nn.Module):
    def __init__(self, inplanes, planes, stride=1, dilation=1, se=False):
        super(ResBlock, self).__init__()
        self.conv1 = torch.nn.Conv2d(inplanes,
                                     planes,
                                     kernel_size=3,
                                     stride=stride,
                                     padding=dilation,
                                     bias=False,
                                     dilation=dilation)
        self.conv1.weight.data.normal_(0.0, 0.04)
        self.relu = torch.nn.ReLU(inplace=True)
        self.conv2 = torch.nn.Conv2d(inplanes,
                                     planes,
                                     kernel_size=3,
                                     stride=stride,
                                     padding=dilation,
                                     bias=False,
                                     dilation=dilation)
        self.conv2.weight.data.normal_(0.0, 0.04)
        self.se = se
        if self.se:
            self.numgroups = 2**(int(np.log2(planes)) // 2)
            self.selayer = SELayer(planes, planes // self.numgroups)

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.relu(out)
        out = self.conv2(out)
        if self.se:
            out = self.selayer(out)
        out += residual
        return out


class SELayer(nn.Module):
    def __init__(self, channel, reduction):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(nn.Linear(channel, reduction),
                                nn.ReLU(inplace=True),
                                nn.Linear(reduction, channel), nn.Sigmoid())

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y


class ConvBlock(nn.Module):
    def __init__(self,
                 input_size,
                 output_size,
                 kernel_size=4,
                 stride=2,
                 dilation=1,
                 padding=1,
                 activation=True,
                 batch_norm=True,
                 dropout=False,
                 BasicBlock=0,
                 se=False):
        super(ConvBlock, self).__init__()
        self.conv = torch.nn.Conv2d(input_size,
                                    output_size,
                                    kernel_size,
                                    stride,
                                    padding,
                                    dilation=dilation)

        self.activation = activation
        if activation:
            self.lrelu = nn.LeakyReLU(0.2, True)

        self.batch_norm = batch_norm
        if batch_norm:
            self.bn = nn.BatchNorm2d(output_size, affine=True)

        self.dropout = dropout
        if dropout:
            self.drop = nn.Dropout(0.5)

        self.BasicBlock = BasicBlock
        if BasicBlock:
            BS = []
            for i in range(BasicBlock):
                BS.append(ResBlock(output_size, output_size, se=se))
            self.Res = nn.Sequential(*BS)

    def forward(self, x):
        if self.activation:
            x = self.lrelu(x)

        x = self.conv(x)
        if self.batch_norm:
            x = self.bn(x)
        if self.dropout:
            x = self.drop(x)
        if self.BasicBlock:
            x = self.Res(x)

        return x


class DeconvBlock(nn.Module):
    def __init__(self,
                 input_size,
                 output_size,
                 kernel_size=4,
                 stride=2,
                 padding=1,
                 batch_norm=True,
                 dropout=False,
                 ups=False,
                 se=False):
        super(DeconvBlock, self).__init__()
        self.ups = ups
        self.se = se
        if self.ups:
            self.deconv = nn.Upsample(scale_factor=2,
                                      mode='bilinear',
                                      align_corners=True)
            self.conv = nn.Conv2d(input_size,
                                  output_size,
                                  kernel_size=3,
                                  padding=1)
        else:
            self.deconv = nn.ConvTranspose2d(input_size, output_size,
                                             kernel_size, stride,
                                             padding)

        if self.se:
            self.numgroups = 2**(int(np.log2(output_size)) // 2)
            self.selayer = SELayer(output_size, output_size // self.numgroups)

        self.relu = nn.ReLU(True)

        self.batch_norm = batch_norm
        if self.batch_norm:
            self.bn = nn.BatchNorm2d(output_size, affine=True)

        self.dropout = dropout
        if self.dropout:
            self.drop = nn.Dropout(0.5)

    def forward(self, x):

        x = self.deconv(self.relu(x))
        if self.ups:
            x = self.conv(x)
        if self.se:
            x = self.selayer(x)
        if self.batch_norm:
            x = self.bn(x)

        if self.dropout:
            x = self.drop(x)
        return x
