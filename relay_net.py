import torch
import torch.nn as nn
import relaynet_pytorch.architecture as arch


class ReLayNet(nn.Module):
    """
    param ={
        'num_channels':1,
        'num_filters':64,
        'num_channels':64,
        'kernel_h':3,
        'kernel_w':3,
        'stride_conv':1,
        'pool':2,
        'stride_pool':2,
        'num_classes':4
    }
    """

    def __init__(self, params):
        super(ReLayNet, self).__init__()

        self.encode1 = arch.EncoderBlock(params)
        params['num_channels'] = 64
        self.encode2 = arch.EncoderBlock(params)
        self.encode3 = arch.EncoderBlock(params)
        self.bottleneck = arch.BasicBlock(params)
        params['num_channels'] = 128
        self.decode1 = arch.DecoderBlock(params)
        self.decode2 = arch.DecoderBlock(params)
        self.decode3 = arch.DecoderBlock(params)
        params['num_channels'] = 64
        self.classifier = arch.ClassifierBlock(params)

    def forward(self, input):
        e1, out1, ind1 = self.encode1.forward(input)
        e2, out2, ind2 = self.encode2.forward(e1)
        e3, out3, ind3 = self.encode3.forward(e2)
        bn = self.bottleneck.forward(e3)

        d3 = self.decode1.forward(bn, out3, ind3)
        d2 = self.decode2.forward(d3, out2, ind2)
        d1 = self.decode3.forward(d2, out1, ind1)
        prob = self.classifier.forward(d1)

        return prob

    @property
    def is_cuda(self):
        return next(self.parameters()).is_cuda

    def save(self, path):
        print('Saving model... %s' % path)
        torch.save(self, path)
        print('Saved.')
