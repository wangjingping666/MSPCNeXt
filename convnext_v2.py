import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import trunc_normal_, DropPath
from timm.models.registry import register_model
from thop import profile

class BN(nn.Module):
    def __init__(self, out_channels,**kwargs):
        super(BN, self).__init__()
        self.bn = nn.BatchNorm2d(out_channels, eps=0.00001)
        self.act = nn.GELU()
    def forward(self, x):
        x = self.bn(x)
        x = self.act(x)
        return x

class Block_3(nn.Module):
    def __init__(self, dim,drop_path=0.6, layer_scale_init_value=1e-6):
        super(Block_3, self).__init__()
        # self.dwconv = nn.Conv2d(dim, dim, kernel_size=3, padding=1, groups=dim)  # depthwise conv
        self.dwconv3_3_a1 = nn.Conv2d(dim, 96, kernel_size=(3,1), padding=(0,1), groups=96)  # depthwise conv
        self.dwconv3_3_a2 = nn.Conv2d(96, 96, kernel_size=(1,3), padding=(1,0), groups=96)  # depthwise conv
        self.bn1 = BN(96, eps=0.00001)
        # self.norm = nn.LayerNorm(dim, eps=1e-6)
        self.pwconv1 = nn.Linear(dim, 4 * dim)
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(4 * dim, dim)
        self.gamma = nn.Parameter(layer_scale_init_value * torch.ones((dim)),
                                  requires_grad=True) if layer_scale_init_value > 0 else None

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        input = x
        # x = self.dwconv(x)
        x = self.dwconv3_3_a1(x)
        x = self.dwconv3_3_a2(x)
        x = self.bn1(x)
        x = x.permute(0, 2, 3, 1)
        # x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        if self.gamma is not None:
            x = self.gamma * x
        x = x.permute(0, 3, 1, 2)
        x = input + self.drop_path(x)
        return x


class Block_5(nn.Module):
    def __init__(self, dim,drop_path=0.6, layer_scale_init_value=1e-6):
        super(Block_5, self).__init__()
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=5, padding=2, groups=dim)  # depthwise conv
        self.norm = nn.LayerNorm(dim, eps=1e-6)
        self.pwconv1 = nn.Linear(dim, 4 * dim)
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(4 * dim, dim)
        self.gamma = nn.Parameter(layer_scale_init_value * torch.ones((dim)),
                                  requires_grad=True) if layer_scale_init_value > 0 else None

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        input = x
        x = self.dwconv(x)
        x = x.permute(0, 2, 3, 1)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        if self.gamma is not None:
            x = self.gamma * x
        x = x.permute(0, 3, 1, 2)
        x = input + self.drop_path(x)
        return x

class Block_7(nn.Module):
    def __init__(self, dim,drop_path=0.6, layer_scale_init_value=1e-6):
        super(Block_7, self).__init__()
        # self.dwconv1_1_7 = nn.Conv2d(dim,96, kernel_size=1)
        # self.dwconv = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim)  # depthwise conv
        self.dwconv7_7_a1 = nn.Conv2d(dim, 96, kernel_size=(7, 1), padding=1, groups=96)
        self.dwconv7_7_a2 = nn.Conv2d(96, 96, kernel_size=(1, 7), padding=2, groups=96)
        self.bn2 = BN(96, eps=0.00001)
        # self.norm = nn.LayerNorm(dim, eps=1e-6)
        self.pwconv1 = nn.Linear(96, 4 * dim)
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(4 * dim, dim)
        self.gamma = nn.Parameter(layer_scale_init_value * torch.ones((dim)),
                                  requires_grad=True) if layer_scale_init_value > 0 else None

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        input = x
        # x = self.dwconv1_1_7(x)
        # x = self.dwconv(x)
        x = self.dwconv7_7_a1(x)
        x = self.dwconv7_7_a2(x)
        x = self.bn2(x)
        x = x.permute(0, 2, 3, 1)
        # x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        if self.gamma is not None:
            x = self.gamma * x
        x = x.permute(0, 3, 1, 2)
        x = input + self.drop_path(x)
        return x





class BNConvBlock_d_1(nn.Module):
    def __init__(self, dim, drop_path=0.6, layer_scale_init_value=1e-6):
        super(BNConvBlock_d_1, self).__init__()
        dim_d = int(dim / 4)
        self.dwconv1_1_1 = nn.Conv2d(dim, dim_d, kernel_size=1)
        self.bn1 = BN(dim_d, eps=0.00001)
                # self.dwconv1_1_3 = nn.Conv2d(dim, dim_d, kernel_size=1)
        #         self.dwconv3_3_a = nn.Conv2d(dim_d, dim_d, kernel_size=3,padding=1,groups=dim_d) #, padding=3, groups=dim_d)
        #         self.bn1 = BN(dim_d, eps=0.00001)

        # self.dwconv1_1_5 = nn.Conv2d(dim, dim_d, kernel_size=1)
        # self.dwconv5_5_a = nn.Conv2d(dim_d, dim_d, kernel_size=5,padding=2,groups=dim_d) #, padding=3, groups=dim_d)
        self.dwconv5_5_a1 = nn.Conv2d(dim, dim_d, kernel_size=(5, 1), padding=1, groups=dim_d)
        self.dwconv5_5_a2 = nn.Conv2d(dim_d, dim_d, kernel_size=(1, 5), padding=1, groups=dim_d)
        self.bn4 = BN(dim_d, eps=0.00001)

        #         self.dwconv1_1_7 = nn.Conv2d(dim, dim_d, kernel_size=1)
        #         # self.dwconv7_7_a = nn.Conv2d(dim_d, dim_d, kernel_size=7, padding=3, groups=dim_d)
        #         # self.bn1 = BN(dim_d, eps=0.001)
        #         self.dwconv7_7_a1 = nn.Conv2d(dim_d, dim_d, kernel_size=(7, 1), padding=1, groups=dim_d)
        #         self.dwconv7_7_a2 = nn.Conv2d(dim_d, dim_d, kernel_size=(1, 7), padding=2, groups=dim_d)
        #         self.bn2 = BN(dim_d, eps=0.00001)

        # self.dwconv1_1_9 = nn.Conv2d(dim, dim_d, kernel_size=1)
        # self.dwconv7_7_a = nn.Conv2d(dim_d, dim_d, kernel_size=7, padding=3, groups=dim_d)
        # self.bn1 = BN(dim_d, eps=0.001)
        self.dwconv9_9_a1 = nn.Conv2d(dim, dim_d, kernel_size=(9, 1), padding=2, groups=dim_d)
        self.dwconv9_9_a2 = nn.Conv2d(dim_d, dim_d, kernel_size=(1, 9), padding=2, groups=dim_d)
        self.bn9 = BN(dim_d, eps=0.00001)

        # self.dwconv1_1_11 = nn.Conv2d(dim, dim_d, kernel_size=1)
        # self.bn3 = BN(dim_d, eps=0.001)
        self.dwconv11_11_b1 = nn.Conv2d(dim, dim_d, kernel_size=(11, 1), padding=2, groups=dim_d)
        # self.bn4 = BN(dim_d, eps=0.001)
        self.dwconv11_11_b2 = nn.Conv2d(dim_d, dim_d, kernel_size=(1, 11), padding=3, groups=dim_d)
        self.bn5 = BN(dim_d, eps=0.00001)

        # #         原始11x11卷积
        #         self.dwconv1_1_11 = nn.Conv2d(dim, dim_d, kernel_size=1)
        #         self.dwconv11_11 = nn.Conv2d(dim_d, dim_d, kernel_size=(11,11), padding=5, groups=dim_d)
        #         self.bn5=BN(dim_d, eps=0.00001)

        #         分解为5个3x3卷积
        # self.dwconv1_1_11 = nn.Conv2d(dim, dim_d, kernel_size=1)
        #         self.dwconv11_11_1 = nn.Conv2d(dim_d, dim_d, kernel_size=(3,3), padding=1, groups=dim_d)
        #         self.dwconv11_11_2= nn.Conv2d(dim_d, dim_d, kernel_size=(3,3), padding=1, groups=dim_d)
        #         self.dwconv11_11_3 = nn.Conv2d(dim_d, dim_d, kernel_size=(3,3), padding=1, groups=dim_d)
        #         self.dwconv11_11_4 = nn.Conv2d(dim_d, dim_d, kernel_size=(3,3), padding=1, groups=dim_d)
        #         self.dwconv11_11_5 = nn.Conv2d(dim_d, dim_d, kernel_size=(3,3), padding=1, groups=dim_d)
        #         self.bn5=BN(dim_d, eps=0.00001)

        #         self.branch_pool = nn.Conv2d(dim , dim_d , kernel_size=1)
        #         self.bn6 = BN(dim_d, eps=0.00001)

        self.pwconv1 = nn.Linear(dim, 4 * dim)
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(4 * dim, dim)
        self.gamma = nn.Parameter(layer_scale_init_value * torch.ones((dim)),
                                  requires_grad=True) if layer_scale_init_value > 0 else None
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        input = x

        dwconv1_1_1 = self.dwconv1_1_1(x)
        dwconv1_1_1 = self.bn1(dwconv1_1_1)
        #         dwconv1_1_3 = self.dwconv1_1_3(x)
        #         dwconv3_3_a = self.dwconv3_3_a(dwconv1_1_3)
        #         dwconv3_3_a1 = self.bn1(dwconv3_3_a)

        # dwconv1_1_5 = self.dwconv1_1_5(x)
        dwconv5_5_a1 = self.dwconv5_5_a1(x)
        dwconv5_5_a2 = self.dwconv5_5_a2(dwconv5_5_a1)
        dwconv5_5_a3 = self.bn4(dwconv5_5_a2)

        #         dwconv1_1_7 = self.dwconv1_1_7(x)
        #         dwconv7_7_a1 = self.dwconv7_7_a1(dwconv1_1_7)
        #         dwconv7_7_a2 = self.dwconv7_7_a2(dwconv7_7_a1)
        #         dwconv7_7_a3 = self.bn2(dwconv7_7_a2)

        # dwconv1_1_9 = self.dwconv1_1_9(x)
        dwconv9_9_a1 = self.dwconv9_9_a1(x)
        dwconv9_9_a2 = self.dwconv9_9_a2(dwconv9_9_a1)
        dwconv9_9_a3 = self.bn9(dwconv9_9_a2)

        # dwconv1_1_11 = self.dwconv1_1_11(x)
        dwconv11_11_b1 = self.dwconv11_11_b1(x)
        dwconv11_11_b3 = self.dwconv11_11_b2(dwconv11_11_b1)
        dwconv11_11_b4 = self.bn5(dwconv11_11_b3)

        #         原始11x11卷积
        #         dwconv1_1_11 = self.dwconv1_1_11(x)
        #         dwconv11_11 = self.dwconv11_11(dwconv1_1_11)
        #         dwconv11_11_b4 = self.bn5(dwconv11_11)

        # 分解为5个3x3卷积
        #         dwconv1_1_11 = self.dwconv1_1_11(x)
        #         dwconv11_11_1 = self.dwconv11_11_1(dwconv1_1_11)
        #         dwconv11_11_2 = self.dwconv11_11_2(dwconv11_11_1)
        #         dwconv11_11_3 = self.dwconv11_11_3(dwconv11_11_2)
        #         dwconv11_11_4 = self.dwconv11_11_4(dwconv11_11_3)
        #         dwconv11_11_5 = self.dwconv11_11_5(dwconv11_11_4)
        #         dwconv11_11_b4 = self.bn5(dwconv11_11_5)

        #         branch_pool = F.avg_pool2d(x, kernel_size=3, stride=1, padding=1)
        #         branch_pool = self.branch_pool(branch_pool)
        #         branch_pool = self.bn6(branch_pool)

        x = [dwconv1_1_1, dwconv5_5_a3, dwconv9_9_a3, dwconv11_11_b4]
        # print(dwconv1_1_1.shape,dwconv7_7_a3.shape, dwconv11_11_b4.shape, branch_pool.shape)
        #         x = [dwconv1_1_1,dwconv3_3_a1,dwconv9_9_a3 ,dwconv11_11_b4]
        #         x = [dwconv3_3_a1,dwconv5_5_a3,dwconv7_7_a3, dwconv11_11_b4]
        #         x = [dwconv1_1_1,dwconv5_5_a3,dwconv7_7_a3, dwconv11_11_b4]
        #         x = [dwconv1_1_1,dwconv3_3_a1,dwconv7_7_a3, dwconv11_11_b4]
        x = torch.cat(x, 1)
        x = x.permute(0, 2, 3, 1)
        # print(x.shape)
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
                    *[BNConvBlock_d_1(dim=dims[i],drop_path=dp_rates[cur + j],
                              layer_scale_init_value=layer_scale_init_value) for j in range(depths[i])]
                )
            else:
                stage = nn.Sequential(
                    *[Block_7(dim=dims[i],drop_path=dp_rates[cur + j],
                              layer_scale_init_value=layer_scale_init_value) for j in range(depths[i])]
                )


            # stage = nn.Sequential(
            #     *[BNConvBlock_d_1(dim=dims[i], drop_path=dp_rates[cur + j],
            #                       layer_scale_init_value=layer_scale_init_value) for j in range(depths[i])]
            # )

            # if i <= 1:
            #     stage = nn.Sequential(
            #         *[Block_7(dim=dims[i],drop_path=dp_rates[cur + j],
            #                   layer_scale_init_value=layer_scale_init_value) for j in range(depths[i])]
            #     )
            # else:
            #     stage = nn.Sequential(
            #         *[BNConvBlock_d_1(dim=dims[i],drop_path=dp_rates[cur + j],
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
            s = (x - u).pow(2).mean(1, keepdim=True)  # 方差
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x


model_urls = {
    "convnext_tiny_1k": "https://dl.fbaipublicfiles.com/convnext/convnext_tiny_1k_224_ema.pth",
}

@register_model
def ConvNeXtPlusv2(pretrained=False, in_22k=False, **kwargs):
    model = ConvNeXt(depths=[3, 3, 9, 3], dims=[96, 192, 384, 768],**kwargs)
    if pretrained:
        url = model_urls['convnext_tiny_22k'] if in_22k else model_urls['convnext_tiny_1k']
        checkpoint = torch.hub.load_state_dict_from_url(url=url, map_location="cpu", check_hash=True)
        model.load_state_dict(checkpoint["model"])
    return model
#
if __name__ == '__main__':
    inputs = torch.randn(1, 3, 224, 224)
    model = ConvNeXtPlusv2()
    print(model)
    # print(summary(model, input_shape=(3, 224, 224)))
    outputs = model(inputs)
    print(outputs)
    flops, params = profile(model, (inputs,))
    print('flops: ', flops, 'params: ', params)
    print("输入维度:", inputs.shape)
    print("输出维度:", outputs.shape)