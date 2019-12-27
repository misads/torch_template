import torch.nn as nn
import numpy as np

from model_zoo.unet import blockUNet


class BEGAN(nn.Module):
    def __init__(self, nc=3, ndf=128, hidden_size=64):
        """
            BEGAN
            https://arxiv.org/pdf/1703.10717.pdf
            :param nc:
            :param ndf:
            :param hidden_size:
        """
        super(BEGAN, self).__init__()

        # 256
        self.conv1 = nn.Sequential(nn.Conv2d(nc, ndf, kernel_size=3, stride=1, padding=1),
                                   nn.ELU(True))
        # 256
        self.conv2 = conv_block(ndf, ndf)
        # 128
        self.conv3 = conv_block(ndf, ndf * 2)
        # 64
        self.conv4 = conv_block(ndf * 2, ndf * 3)
        # 32

        # self.conv5 = conv_block(ndf * 3, ndf * 4)
        # 16
        # self.conv6 = conv_block(ndf * 4, ndf * 5)

        self.encode = nn.Conv2d(ndf * 3, hidden_size, kernel_size=1, stride=1, padding=0)
        self.decode = nn.Conv2d(hidden_size, ndf, kernel_size=1, stride=1, padding=0)
        # 32
        self.deconv4 = deconv_block(ndf, ndf)
        # 64
        self.deconv3 = deconv_block(ndf, ndf)
        # 128
        self.deconv2 = deconv_block(ndf, ndf)
        # 256
        self.deconv1 = nn.Sequential(nn.Conv2d(ndf, ndf, kernel_size=3, stride=1, padding=1),
                                     nn.ELU(True),
                                     nn.Conv2d(ndf, ndf, kernel_size=3, stride=1, padding=1),
                                     nn.ELU(True),
                                     nn.Conv2d(ndf, nc, kernel_size=3, stride=1, padding=1),
                                     nn.Tanh())
        """
        self.deconv1 = nn.Sequential(nn.Conv2d(ndf,nc,kernel_size=3,stride=1,padding=1),
                                     nn.Tanh())
        """

    def forward(self, x):
        out1 = self.conv1(x)
        out2 = self.conv2(out1)
        out3 = self.conv3(out2)
        out4 = self.conv4(out3)
        out5 = self.encode(out4)
        dout5 = self.decode(out5)
        dout4 = self.deconv4(dout5)
        dout3 = self.deconv3(dout4)
        dout2 = self.deconv2(dout3)
        dout1 = self.deconv1(dout2)
        return dout1


def conv_block(in_dim, out_dim):
    return nn.Sequential(nn.Conv2d(in_dim, in_dim, kernel_size=3, stride=1, padding=1),
                         nn.ELU(True),
                         nn.Conv2d(in_dim, in_dim, kernel_size=3, stride=1, padding=1),
                         nn.ELU(True),
                         nn.Conv2d(in_dim, out_dim, kernel_size=1, stride=1, padding=0),
                         nn.AvgPool2d(kernel_size=2, stride=2))


def deconv_block(in_dim, out_dim):
    return nn.Sequential(nn.Conv2d(in_dim, out_dim, kernel_size=3, stride=1, padding=1),
                         nn.ELU(True),
                         nn.Conv2d(out_dim, out_dim, kernel_size=3, stride=1, padding=1),
                         nn.ELU(True),
                         nn.UpsamplingNearest2d(scale_factor=2))

########################################################
########################################################


class UNet_Encoder(nn.Module):
    def __init__(self, nc=3, nf=128):
        super(UNet_Encoder, self).__init__()

        main = nn.Sequential()
        # 256
        layer_idx = 1
        name = 'layer%d' % layer_idx
        main.add_module('%s_conv' % name, nn.Conv2d(nc, nf, 4, 2, 1, bias=False))

        # 128
        layer_idx += 1
        name = 'layer%d' % layer_idx
        main.add_module(name, blockUNet(nf, nf * 2, name, transposed=False, bn=True, relu=False, dropout=False))

        # 64
        layer_idx += 1
        name = 'layer%d' % layer_idx
        nf = nf * 2
        main.add_module(name, blockUNet(nf, nf * 2, name, transposed=False, bn=True, relu=False, dropout=False))

        # 32
        layer_idx += 1
        name = 'layer%d' % layer_idx
        nf = nf * 2
        main.add_module('%s_leakyrelu' % name, nn.LeakyReLU(0.2, inplace=True))
        main.add_module('%s_conv' % name, nn.Conv2d(nf, nf * 2, 4, 1, 1, bias=False))
        main.add_module('%s_bn' % name, nn.BatchNorm2d(nf * 2))

        # 31
        layer_idx += 1
        name = 'layer%d' % layer_idx
        nf = nf * 2
        main.add_module('%s_leakyrelu' % name, nn.LeakyReLU(0.2, inplace=True))
        main.add_module('%s_conv' % name, nn.Conv2d(nf, 1, 4, 1, 1, bias=False))
        main.add_module('%s_sigmoid' % name, nn.Sigmoid())
        # 30 (sizePatchGAN=30)

        self.main = main

    def forward(self, x):
        output = self.main(x)
        return output

########################################################
########################################################


class MultiscaleDiscriminator(nn.Module):
    def __init__(self, input_nc=3, ndf=64, n_layers=3, norm_layer=nn.BatchNorm2d,
                 use_sigmoid=False, num_D=3, getIntermFeat=False):
        """
            MultiscaleDiscriminator of Pix2pixHD
            :param input_nc:
            :param ndf:
            :param n_layers:
            :param norm_layer:
            :param use_sigmoid:
            :param num_D:
            :param getIntermFeat:
        """
        super(MultiscaleDiscriminator, self).__init__()
        self.num_D = num_D
        self.n_layers = n_layers
        self.getIntermFeat = getIntermFeat

        for i in range(num_D):
            netD = NLayerDiscriminator(input_nc, ndf, n_layers, norm_layer, use_sigmoid, getIntermFeat)
            if getIntermFeat:
                for j in range(n_layers + 2):
                    setattr(self, 'scale' + str(i) + '_layer' + str(j), getattr(netD, 'model' + str(j)))
            else:
                setattr(self, 'layer' + str(i), netD.model)

        self.downsample = nn.AvgPool2d(3, stride=2, padding=[1, 1], count_include_pad=False)

    def singleD_forward(self, model, input):
        if self.getIntermFeat:
            result = [input]
            for i in range(len(model)):
                result.append(model[i](result[-1]))
            return result[1:]
        else:
            return [model(input)]

    def forward(self, input):
        num_D = self.num_D
        result = []
        input_downsampled = input
        for i in range(num_D):
            if self.getIntermFeat:
                model = [getattr(self, 'scale' + str(num_D - 1 - i) + '_layer' + str(j)) for j in
                         range(self.n_layers + 2)]
            else:
                model = getattr(self, 'layer' + str(num_D - 1 - i))
            result.append(self.singleD_forward(model, input_downsampled))
            if i != (num_D - 1):
                input_downsampled = self.downsample(input_downsampled)
        return result

########################################################
########################################################

# Defines the PatchGAN discriminator with the specified arguments.
class NLayerDiscriminator(nn.Module):
    def __init__(self, input_nc=3, ndf=64, n_layers=3, norm_layer=nn.BatchNorm2d, use_sigmoid=False, getIntermFeat=False):
        """
            PatchGAN discriminator
            :param input_nc:
            :param ndf:
            :param n_layers:
            :param norm_layer:
            :param use_sigmoid:
            :param getIntermFeat:
        """
        super(NLayerDiscriminator, self).__init__()
        self.getIntermFeat = getIntermFeat
        self.n_layers = n_layers

        kw = 4
        padw = int(np.ceil((kw - 1.0) / 2))
        sequence = [[nn.Conv2d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw), nn.LeakyReLU(0.2, True)]]

        nf = ndf
        for n in range(1, n_layers):
            nf_prev = nf
            nf = min(nf * 2, 512)
            sequence += [[
                nn.Conv2d(nf_prev, nf, kernel_size=kw, stride=2, padding=padw),
                norm_layer(nf), nn.LeakyReLU(0.2, True)
            ]]

        nf_prev = nf
        nf = min(nf * 2, 512)
        sequence += [[
            nn.Conv2d(nf_prev, nf, kernel_size=kw, stride=1, padding=padw),
            norm_layer(nf),
            nn.LeakyReLU(0.2, True)
        ]]

        sequence += [[nn.Conv2d(nf, 1, kernel_size=kw, stride=1, padding=padw)]]

        if use_sigmoid:
            sequence += [[nn.Sigmoid()]]

        if getIntermFeat:
            for n in range(len(sequence)):
                setattr(self, 'model' + str(n), nn.Sequential(*sequence[n]))
        else:
            sequence_stream = []
            for n in range(len(sequence)):
                sequence_stream += sequence[n]
            self.model = nn.Sequential(*sequence_stream)

    def forward(self, input):
        if self.getIntermFeat:
            res = [input]
            for n in range(self.n_layers + 2):
                model = getattr(self, 'model' + str(n))
                res.append(model(res[-1]))
            return res[1:]
        else:
            return self.model(input)