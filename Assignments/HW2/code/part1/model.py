from torch import nn


class ConvNet(nn.Module):
    """CNN for SVNH dataset"""

    def __init__(self, in_dim, out_dim):
        super(ConvNet, self).__init__()
        channels_list = [128]
        in_channel_list = [in_dim] + channels_list[:]
        out_channel_list = channels_list[:] + [out_dim]
        conv_layer = []
        print(f'in: {in_channel_list}, out: {out_channel_list}')
        for in_channel, out_channel in zip(in_channel_list, out_channel_list):
            conv_layer.append(ResBlock(in_channels=in_channel, out_channels=out_channel))
            print(f'in channel: {in_channel}, out_channel: {out_channel}')
        self.conv_layer = nn.Sequential(*conv_layer)

        self.fc_layer = nn.Sequential(
            nn.Dropout(0.8),
            nn.Linear(16384, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.4),
            nn.Linear(128, 10),
        )

    def forward(self, x):
        x = self.conv_layer(x)
        x = x.view(x.size(0), -1)
        y = self.fc_layer(x)
        # print(f'y: {y}')
        return y


class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ResBlock, self).__init__()
        self.main_block = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout(p=0.2)
        )
        shortcut_layers = []
        if in_channels != out_channels:
            shortcut_layers.append(nn.Conv2d(in_channels=in_channels,
                                   out_channels=out_channels,
                                   kernel_size=1,
                                   stride=2,
                                   bias=False)
        )
        self.shortcut_path = nn.Sequential(*shortcut_layers)

    def forward(self, x):
        main_path = self.main_block(x)
        # print(f'main_path shape: {main_path.shape}')
        residual = self.shortcut_path(x)
        # print(f'residual shape: {residual.shape}')
        res = nn.functional.relu(main_path + residual)
        return res
