import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import trunc_normal_, DropPath
from timm.models.registry import register_model
from thop import profile





# class Block_5(nn.Module):
#     def __init__(self, dim,drop_path=0.6, layer_scale_init_value=1e-6):
#         super(Block_5, self).__init__()
#         self.dwconv5_5_a1 = nn.Conv2d(dim, 96, kernel_size=(5,1), padding=1, groups=96)
#         self.dwconv5_5_a2 = nn.Conv2d(96, 96, kernel_size=(1,5), padding=1, groups=96)
#         self.bn2 = BN(96, eps=0.00001)
#         self.pwconv1 = nn.Linear(dim, 4 * dim)
#         self.act = nn.GELU()
#         self.pwconv2 = nn.Linear(4 * dim, dim)
#         self.gamma = nn.Parameter(layer_scale_init_value * torch.ones((dim)),
#                                   requires_grad=True) if layer_scale_init_value > 0 else None
#
#         self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
#
#     def forward(self, x):
#         input = x
#         x = self.dwconv5_5_a1(x)
#         x = self.dwconv5_5_a2(x)
#         x = self.bn2(x)
#         x = x.permute(0, 2, 3, 1)
#         x = self.pwconv1(x)
#         x = self.act(x)
#         x = self.pwconv2(x)
#         if self.gamma is not None:
#             x = self.gamma * x
#         x = x.permute(0, 3, 1, 2)
#         x = input + self.drop_path(x)
#         return x

"""
The 7*7 convolution block for Stage1. Adapted from ConvNeXt_Block. Factorise the 7*7 convolution kernel in the original ConvNeXt_Block.
"""
class Block_7(nn.Module):
    def __init__(self, dim,drop_path=0.6, layer_scale_init_value=1e-6):
        super(Block_7, self).__init__()
        self.dwconv7_7_a1 = nn.Conv2d(dim, 96, kernel_size=(7, 1), padding=1, groups=96)
        self.dwconv7_7_a2 = nn.Conv2d(96, 96, kernel_size=(1, 7), padding=2, groups=96)
        self.bn2 = BN(96, eps=0.00001)    #Each branch is immediately followed by a BN module.
        self.pwconv1 = nn.Linear(96, 4 * dim)
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(4 * dim, dim)
        self.gamma = nn.Parameter(layer_scale_init_value * torch.ones((dim)),
                                  requires_grad=True) if layer_scale_init_value > 0 else None

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        input = x
        x = self.dwconv7_7_a1(x)
        x = self.dwconv7_7_a2(x)
        x = self.bn2(x)
        x = x.permute(0, 2, 3, 1)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        if self.gamma is not None:
            x = self.gamma * x
        x = x.permute(0, 3, 1, 2)
        x = input + self.drop_path(x)
        return x



class ConvNeXt(nn.Module):
    def __init__(self, in_chans=3, num_classes=7, depths=[3, 3, 9, 3], dims=[96, 192, 384, 768],
                 drop_path_rate=0.6, layer_scale_init_value=1e-6, head_init_scale=1.):
        super(ConvNeXt, self).__init__()

        self.downsample_layers = nn.ModuleList()
        stem = nn.Sequential(
            nn.Conv2d(in_chans, dims[0], kernel_size=4, stride=4),
            LayerNorm(dims[0], eps=1e-6, data_format="channels_first")
        )
        self.downsample_layers.append(stem)

        for i in range(3):
            downsample_layer = nn.Sequential(
                LayerNorm(dims[i], eps=1e-6,data_format="channels_first"),
                nn.Conv2d(dims[i], dims[i + 1], kernel_size=2, stride=2)
            )
            self.downsample_layers.append(downsample_layer)

        self.stages = nn.ModuleList()
        dp_rates = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        cur = 0


        """
        Use customised Block modules in each Stage.
        """
        for i in range(4):
            if i == 0:
                stage = nn.Sequential(
                    *[Block_3(dim=dims[i], drop_path=dp_rates[cur + j],
                            layer_scale_init_value=layer_scale_init_value) for j in range(depths[i])]
                )
            elif i == 1:
                 stage = nn.Sequential(
                    *[Block_7(dim=dims[i], drop_path=dp_rates[cur + j],
                                  layer_scale_init_value=layer_scale_init_value) for j in range(depths[i])]
                )
            elif i == 2:
                stage = nn.Sequential(
                    *[mspcnext_v2_Block(dim=dims[i],drop_path=dp_rates[cur + j],
                              layer_scale_init_value=layer_scale_init_value) for j in range(depths[i])]
                )
            else:
                stage = nn.Sequential(
                    *[Block_7(dim=dims[i],drop_path=dp_rates[cur + j],
                              layer_scale_init_value=layer_scale_init_value) for j in range(depths[i])]
                )

            """
            The 4 Stages use the same mspcnext_v2_Block.
            """
            # stage = nn.Sequential(
            #     *[mspcnext_v2_Block(dim=dims[i], drop_path=dp_rates[cur + j],
            #                       layer_scale_init_value=layer_scale_init_value) for j in range(depths[i])]
            # )

            """
            The first two Stages use a custom 7*7 convolution module. The last two Stages use mspcnext_v2_Block.
            """
            # if i <= 1:
            #     stage = nn.Sequential(
            #         *[Block_7(dim=dims[i],drop_path=dp_rates[cur + j],
            #                   layer_scale_init_value=layer_scale_init_value) for j in range(depths[i])]
            #     )
            # else:
            #     stage = nn.Sequential(
            #         *[mspcnext_v2_Block(dim=dims[i],drop_path=dp_rates[cur + j],
            #                   layer_scale_init_value=layer_scale_init_value) for j in range(depths[i])]
            #     )


            self.stages.append(stage)
            cur += depths[i]

        self.norm = nn.LayerNorm(dims[-1], eps=1e-6)
        self.head = nn.Linear(dims[-1], num_classes)

        self.apply(self._init_weights)
        self.head.weight.data.mul_(head_init_scale)
        self.head.bias.data.mul_(head_init_scale)

    def _init_weights(self, m):
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            nn.init.trunc_normal_(m.weight, std=0.02)
            nn.init.constant_(m.bias, 0)

    def forward_features(self, x):
        for i in range(4):
            x = self.downsample_layers[i](x)
            x = self.stages[i](x)
        return self.norm(x.mean([-2, -1]))

    def forward(self, x):
        x = self.forward_features(x)
        x = self.head(x)
        return x


class LayerNorm(nn.Module):
    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError
        self.normalized_shape = (normalized_shape,)

    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x


model_urls = {
    "convnext_tiny_1k": "https://dl.fbaipublicfiles.com/convnext/convnext_tiny_1k_224_ema.pth",
}

@register_model
def mspcnext_v2(pretrained=False, in_22k=False, **kwargs):
    model = ConvNeXt(depths=[3, 3, 9, 3], dims=[96, 192, 384, 768],**kwargs)
    if pretrained:
        url = model_urls['convnext_tiny_22k'] if in_22k else model_urls['convnext_tiny_1k']
        checkpoint = torch.hub.load_state_dict_from_url(url=url, map_location="cpu", check_hash=True)
        model.load_state_dict(checkpoint["model"])
    return model

if __name__ == '__main__':
    inputs = torch.randn(1, 3, 224, 224)
    model = mspcnext_v2()
    print(model)
    outputs = model(inputs)
    print(outputs)
    flops, params = profile(model, (inputs,))
    print('flops: ', flops, 'params: ', params)
    print("input dimension:", inputs.shape)
    print("output dimension:", outputs.shape)
