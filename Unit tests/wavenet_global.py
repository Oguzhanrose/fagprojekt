import torch
import torch.nn as nn
import torch.nn.functional as F

class Wavenet(nn.Module):
    def __init__(self, quantization_bins, channels, dilation_depth, blocks):
        super(Wavenet, self).__init__()

        """ Part 1: Define model parameters """""
        self.C = channels
        self.kernel_size = 2
        self.bins = quantization_bins
        self.dilations = [2 ** i for i in range(dilation_depth)] * blocks

        """ Part 2: Define model layers """
        self.pre_process_conv = nn.Conv1d(in_channels=self.bins, out_channels=self.C, kernel_size=1)
        self.causal_layers = nn.ModuleList()

        for d in self.dilations:
            self.causal_layers.append(ResidalLayer(in_channels=self.C, out_channels=self.C, dilation=d, kernel_size=self.kernel_size))

        self.post_process_conv1 = nn.Conv1d(self.C, self.C, kernel_size=1)
        self.post_process_conv2 = nn.Conv1d(self.C, self.bins, kernel_size=1)


    def forward(self, x, gc):
        """ Function: Makes the forward pass/model prediction
            Input: Mu- and one-hot-encoded waveform. The shape of the input is (batch_size, quantization_bins, samples).
                   It is important that 'x' has at least the length of the models receptive field.
            Output: Distribution for prediction of next sample. Shape (batch_size, quantization_bins, what's left after
                    dilation, should be 1 at inference) """

        """ Part 1: Through pre-processing layer """
        x = self.pre_process_conv(x)

        """ Part 2: Through stack of dilated causal convolutions """
        skips, skip = [], None

        for layer in self.causal_layers:
            x, skip = layer(x, gc)

            # Save skip connection results
            skips.append(skip)

        """ Part 3: Post processes (-softmax) """
        # Add skip connections together
        x = sum([s[:, :, -skip.size(2):] for s in skips])

        # Do the rest of the preprocessing 
        x = F.relu(x)
        x = self.post_process_conv1(x)  # shape --> (batch_size, channels, samples)
        x = F.relu(x)
        x = self.post_process_conv2(x)  # shape --> (batch_size, quantization_bins, samples)

        return x


class ResidalLayer(nn.Module):
    """ Class description: This class is a sub-model of a residual layer (see research paper)"""

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, dilation: int):
        super(ResidalLayer, self).__init__()

        """ Part 1: Define model parameters """
        self.dilation = dilation

        """ Part 2: Define model layers """
        # The original Wa original WaveNet paper used a single shared 1x1 conv for both filter (f) and gate (g).
        # Instead we use one for each here i.e. conv_f and conv_g.
        self.conv_f = nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size, dilation=dilation)
        self.conv_g = nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size, dilation=dilation)

        # Global conditioning matrix-vector product layer
        self.gc_layer_f = nn.Linear(10, out_channels)
        self.gc_layer_g = nn.Linear(10, out_channels)

        # 1 shared 1x1 convolution
        self.conv_1x1 = nn.Conv1d(out_channels, out_channels, kernel_size=1)

    def forward(self, x, gc):
        """ Function: Do forward pass/make model prediction """
        # Convolutions
        f_x = self.conv_f(x)
        g_x = self.conv_g(x)

        # Global conditioning matrix-vector product
        f_gc = self.gc_layer_f(gc).view(x.size(0), -1, 1)
        g_gc = self.gc_layer_g(gc).view(x.size(0), -1, 1)

        # Send through gate
        f = torch.tanh(f_x + f_gc)
        g = torch.sigmoid(g_x + g_gc)
        z = f * g

        # Save for skip connection
        skip = self.conv_1x1(z)  # torch shape (10x32x99)

        # Save residual as input to next layer residual layer
        residual = x[:, :, self.dilation:] + skip

        return residual, skip
