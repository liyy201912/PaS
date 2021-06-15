import torch.nn as nn
import numpy as np
import torch
from resnet import BinaryConv2d


def conv_bn(in_channels, out_channels, kernel_size, stride, padding, groups=1):
    result = nn.Sequential()
    result.add_module('conv', nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
                                        kernel_size=kernel_size, stride=stride, padding=padding, groups=groups,
                                        bias=False))
    result.add_module('bn', nn.BatchNorm2d(num_features=out_channels))
    return result


class RepVGGBlock(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size,
                 stride=1, padding=0, dilation=1, groups=1, padding_mode='zeros', deploy=False):
        super(RepVGGBlock, self).__init__()
        self.deploy = deploy
        self.groups = groups
        self.in_channels = in_channels

        assert kernel_size == 3
        assert padding == 1

        padding_11 = padding - kernel_size // 2

        self.nonlinearity = nn.ReLU()
        # self.scale = BinaryConv2d(out_channels, out_channels, kernel_size=1, stride=1, padding=0,
        #                           groups=out_channels, bias=False)

        if deploy:
            self.rbr_reparam = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                                         stride=stride,
                                         padding=padding, dilation=dilation, groups=groups, bias=True,
                                         padding_mode=padding_mode)

        else:
            self.rbr_identity = nn.BatchNorm2d(
                num_features=in_channels) if out_channels == in_channels and stride == 1 else None
            self.rbr_dense = conv_bn(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                                     stride=stride, padding=padding, groups=groups)
            self.rbr_1x1 = conv_bn(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=stride,
                                   padding=padding_11, groups=groups)
            self.scale = BinaryConv2d(out_channels, out_channels, kernel_size=1, stride=1, padding=0,
                                      groups=out_channels, bias=False)
            # print('RepVGG Block, identity = ', self.rbr_identity)

    def forward(self, inputs):
        if hasattr(self, 'rbr_reparam'):
            return self.nonlinearity(self.rbr_reparam(inputs))

        if self.rbr_identity is None:
            id_out = 0
        else:
            id_out = self.rbr_identity(inputs)

        return self.scale(self.nonlinearity(self.rbr_dense(inputs) + self.rbr_1x1(inputs) + id_out))

    #   This func derives the equivalent kernel and bias in a DIFFERENTIABLE way.
    #   You can get the equivalent kernel and bias at any time and do whatever you want,
    #   for example, apply some penalties or constraints during training, just like you do to the other models.
    #   May be useful for quantization or pruning.
    def get_equivalent_kernel_bias(self):
        kernel3x3, bias3x3 = self._fuse_bn_tensor(self.rbr_dense)
        kernel1x1, bias1x1 = self._fuse_bn_tensor(self.rbr_1x1)
        kernelid, biasid = self._fuse_bn_tensor(self.rbr_identity)
        return kernel3x3 + self._pad_1x1_to_3x3_tensor(kernel1x1) + kernelid, bias3x3 + bias1x1 + biasid

    def _pad_1x1_to_3x3_tensor(self, kernel1x1):
        if kernel1x1 is None:
            return 0
        else:
            return torch.nn.functional.pad(kernel1x1, [1, 1, 1, 1])

    def _fuse_bn_tensor(self, branch):
        if branch is None:
            return 0, 0
        if isinstance(branch, nn.Sequential):
            kernel = branch.conv.weight
            running_mean = branch.bn.running_mean
            running_var = branch.bn.running_var
            gamma = branch.bn.weight
            beta = branch.bn.bias
            eps = branch.bn.eps
        else:
            assert isinstance(branch, nn.BatchNorm2d)
            if not hasattr(self, 'id_tensor'):
                input_dim = self.in_channels // self.groups
                kernel_value = np.zeros((self.in_channels, input_dim, 3, 3), dtype=np.float32)
                for i in range(self.in_channels):
                    kernel_value[i, i % input_dim, 1, 1] = 1
                self.id_tensor = torch.from_numpy(kernel_value).to(branch.weight.device)
            kernel = self.id_tensor
            running_mean = branch.running_mean
            running_var = branch.running_var
            gamma = branch.weight
            beta = branch.bias
            eps = branch.eps
        std = (running_var + eps).sqrt()
        t = (gamma / std).reshape(-1, 1, 1, 1)
        return kernel * t, beta - running_mean * gamma / std

    def repvgg_convert(self):
        kernel, bias = self.get_equivalent_kernel_bias()
        return kernel.detach().cpu().numpy(), bias.detach().cpu().numpy()

    def switch_to_deploy(self):
        if hasattr(self, 'rbr_reparam'):
            return
        kernel, bias = self.get_equivalent_kernel_bias()
        self.rbr_reparam = nn.Conv2d(in_channels=self.rbr_dense.conv.in_channels,
                                     out_channels=self.rbr_dense.conv.out_channels,
                                     kernel_size=self.rbr_dense.conv.kernel_size, stride=self.rbr_dense.conv.stride,
                                     padding=self.rbr_dense.conv.padding, dilation=self.rbr_dense.conv.dilation,
                                     groups=self.rbr_dense.conv.groups, bias=True)
        self.rbr_reparam.weight.data = kernel
        self.rbr_reparam.bias.data = bias
        for para in self.parameters():
            para.detach_()
        self.__delattr__('rbr_dense')
        self.__delattr__('rbr_1x1')
        if hasattr(self, 'rbr_identity'):
            self.__delattr__('rbr_identity')


class RepVGG(nn.Module):

    def __init__(self, num_blocks, num_classes=1000, width_multiplier=None, override_groups_map=None, deploy=False):
        super(RepVGG, self).__init__()

        assert len(width_multiplier) == 4

        self.deploy = deploy
        self.override_groups_map = override_groups_map or dict()

        assert 0 not in self.override_groups_map

        self.in_planes = min(64, int(64 * width_multiplier[0]))

        self.stage0 = RepVGGBlock(in_channels=3, out_channels=self.in_planes, kernel_size=3, stride=2, padding=1,
                                  deploy=self.deploy)
        self.cur_layer_idx = 1
        self.stage1 = self._make_stage(int(64 * width_multiplier[0]), num_blocks[0], stride=2)
        self.stage2 = self._make_stage(int(128 * width_multiplier[1]), num_blocks[1], stride=2)
        self.stage3 = self._make_stage(int(256 * width_multiplier[2]), num_blocks[2], stride=2)
        self.stage4 = self._make_stage(int(512 * width_multiplier[3]), num_blocks[3], stride=2)
        self.gap = nn.AdaptiveAvgPool2d(output_size=1)
        self.linear = nn.Linear(int(512 * width_multiplier[3]), num_classes)

        if not self.deploy:
            for m in self.modules():
                if isinstance(m, RepVGGBlock):
                    nn.init.constant_(m.scale.weight, 1)

    def _make_stage(self, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        blocks = []
        for stride in strides:
            cur_groups = self.override_groups_map.get(self.cur_layer_idx, 1)
            blocks.append(RepVGGBlock(in_channels=self.in_planes, out_channels=planes, kernel_size=3,
                                      stride=stride, padding=1, groups=cur_groups, deploy=self.deploy))
            self.in_planes = planes
            self.cur_layer_idx += 1
        return nn.Sequential(*blocks)

    def forward(self, x):
        out = self.stage0(x)
        out = self.stage1(out)
        out = self.stage2(out)
        out = self.stage3(out)
        out = self.stage4(out)
        out = self.gap(out)
        out = out.reshape(out.size(0), -1)
        out = self.linear(out)
        return out


class NewRepVGG(nn.Module):

    def __init__(self, num_blocks, num_classes=1000, config=None, override_groups_map=None, deploy=False):
        super(NewRepVGG, self).__init__()

        # assert len(width_multiplier) == 4

        self.deploy = deploy
        self.override_groups_map = override_groups_map or dict()

        assert 0 not in self.override_groups_map

        self.in_planes = config[0][0]

        self.stage0 = RepVGGBlock(in_channels=3, out_channels=self.in_planes, kernel_size=3, stride=2, padding=1,
                                  deploy=self.deploy)
        self.cur_layer_idx = 1
        self.stage1 = self._make_stage(config[1], len(config[1]), stride=2)
        self.stage2 = self._make_stage(config[2], len(config[2]), stride=2)
        self.stage3 = self._make_stage(config[3], len(config[3]), stride=2)
        self.stage4 = self._make_stage(config[4], len(config[4]), stride=2)
        self.gap = nn.AdaptiveAvgPool2d(output_size=1)
        self.linear = nn.Linear(config[4][-1], num_classes)

        # for m in self.modules():
        #     if isinstance(m, RepVGGBlock):
        #         nn.init.constant_(m.scale.weight, 1)

    def _make_stage(self, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        blocks = []
        for i in range(len(strides)):
            cur_groups = self.override_groups_map.get(self.cur_layer_idx, 1)
            blocks.append(RepVGGBlock(in_channels=self.in_planes, out_channels=planes[i], kernel_size=3,
                                      stride=strides[i], padding=1, groups=cur_groups, deploy=self.deploy))
            self.in_planes = planes[i]
            self.cur_layer_idx += 1
        return nn.Sequential(*blocks)

    def forward(self, x):
        out = self.stage0(x)
        out = self.stage1(out)
        out = self.stage2(out)
        out = self.stage3(out)
        out = self.stage4(out)
        out = self.gap(out)
        out = out.reshape(out.size(0), -1)
        out = self.linear(out)
        return out


optional_groupwise_layers = [2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26]
g2_map = {l: 2 for l in optional_groupwise_layers}
g4_map = {l: 4 for l in optional_groupwise_layers}


def create_RepVGG_A0(deploy=False):
    return RepVGG(num_blocks=[2, 4, 14, 1], num_classes=1000,
                  width_multiplier=[0.75, 0.75, 0.75, 2.5], override_groups_map=None, deploy=deploy)


def create_RepVGG_A1(deploy=False):
    return RepVGG(num_blocks=[2, 4, 14, 1], num_classes=1000,
                  width_multiplier=[1, 1, 1, 2.5], override_groups_map=None, deploy=deploy)


def create_RepVGG_A2(deploy=False):
    return RepVGG(num_blocks=[2, 4, 14, 1], num_classes=1000,
                  width_multiplier=[1.5, 1.5, 1.5, 2.75], override_groups_map=None, deploy=deploy)


def create_RepVGG_B0(deploy=False):
    return RepVGG(num_blocks=[4, 6, 16, 1], num_classes=1000,
                  width_multiplier=[1, 1, 1, 2.5], override_groups_map=None, deploy=deploy)


def create_RepVGG_B1(deploy=False):
    return RepVGG(num_blocks=[4, 6, 16, 1], num_classes=1000,
                  width_multiplier=[2, 2, 2, 4], override_groups_map=None, deploy=deploy)


def create_RepVGG_B1g2(deploy=False):
    return RepVGG(num_blocks=[4, 6, 16, 1], num_classes=1000,
                  width_multiplier=[2, 2, 2, 4], override_groups_map=g2_map, deploy=deploy)


def create_RepVGG_B1g4(deploy=False):
    return RepVGG(num_blocks=[4, 6, 16, 1], num_classes=1000,
                  width_multiplier=[2, 2, 2, 4], override_groups_map=g4_map, deploy=deploy)


def create_RepVGG_B2(deploy=False):
    return RepVGG(num_blocks=[4, 6, 16, 1], num_classes=1000,
                  width_multiplier=[2.5, 2.5, 2.5, 5], override_groups_map=None, deploy=deploy)


def create_RepVGG_B2g2(deploy=False):
    return RepVGG(num_blocks=[4, 6, 16, 1], num_classes=1000,
                  width_multiplier=[2.5, 2.5, 2.5, 5], override_groups_map=g2_map, deploy=deploy)


def create_RepVGG_B2g4(deploy=False):
    return RepVGG(num_blocks=[4, 6, 16, 1], num_classes=1000,
                  width_multiplier=[2.5, 2.5, 2.5, 5], override_groups_map=g4_map, deploy=deploy)


def create_RepVGG_B3(deploy=False):
    return RepVGG(num_blocks=[4, 6, 16, 1], num_classes=1000,
                  width_multiplier=[3, 3, 3, 5], override_groups_map=None, deploy=deploy)


def create_RepVGG_B3g2(deploy=False):
    return RepVGG(num_blocks=[4, 6, 16, 1], num_classes=1000,
                  width_multiplier=[3, 3, 3, 5], override_groups_map=g2_map, deploy=deploy)


def create_RepVGG_B3g4(deploy=False):
    return RepVGG(num_blocks=[4, 6, 16, 1], num_classes=1000,
                  width_multiplier=[3, 3, 3, 5], override_groups_map=g4_map, deploy=deploy)


def create_RepVGG_config(config=None, deploy=True):
    return NewRepVGG(num_blocks=[4, 6, 16, 1], num_classes=1000,
                     config=[[8], [20, 16, 15, 10], [49, 41, 36, 34, 39, 31],
                             [125, 107, 102, 97, 106, 123, 127, 170, 196, 202, 232,
                              269, 315, 425, 475, 450], [2048]],
                     override_groups_map=None, deploy=deploy)
    # return NewRepVGG(num_blocks=[4, 6, 16, 1], num_classes=1000,
    #                  config=[[11], [29, 32, 27, 23], [68, 57, 55, 65, 79, 79],
    #                          [208, 198, 193, 175, 201, 240, 254, 323, 357, 378, 412, 442, 483, 496, 510, 487],
    #                          [2048]],
    #                  override_groups_map=None, deploy=deploy)


func_dict = {
    'RepVGG-A0': create_RepVGG_A0,
    'RepVGG-A1': create_RepVGG_A1,
    'RepVGG-A2': create_RepVGG_A2,
    'RepVGG-B0': create_RepVGG_B0,
    'RepVGG-B1': create_RepVGG_B1,
    'RepVGG-B1g2': create_RepVGG_B1g2,
    'RepVGG-B1g4': create_RepVGG_B1g4,
    'RepVGG-B2': create_RepVGG_B2,
    'RepVGG-B2g2': create_RepVGG_B2g2,
    'RepVGG-B2g4': create_RepVGG_B2g4,
    'RepVGG-B3': create_RepVGG_B3,
    'RepVGG-B3g2': create_RepVGG_B3g2,
    'RepVGG-B3g4': create_RepVGG_B3g4,
    'RepVGG-config': create_RepVGG_config,
}


def get_RepVGG_func_by_name(name):
    return func_dict[name]


#   Use this for converting a customized model with RepVGG as one of its components
#   (e.g., the backbone of a semantic segmentation model)
#   The use case will be like
#   1.  Build train_model. For example, build a PSPNet with a training-time RepVGG as backbone
#   2.  Train train_model or do whatever you want
#   3.  Build deploy_model. In the above example, that will be a PSPNet with an inference-time RepVGG as backbone
#   4.  Call this func
#   ====================== the pseudo code will be like
#   train_backbone = create_RepVGG_B2(deploy=False)
#   train_backbone.load_state_dict(torch.load('RepVGG-B2-train.pth'))
#   train_pspnet = build_pspnet(backbone=train_backbone)
#   segmentation_train(train_pspnet)
#   deploy_backbone = create_RepVGG_B2(deploy=True)
#   deploy_pspnet = build_pspnet(backbone=deploy_backbone)
#   whole_model_convert(train_pspnet, deploy_pspnet)
#   segmentation_test(deploy_pspnet)
def whole_model_convert(train_model: torch.nn.Module, deploy_model: torch.nn.Module, save_path=None):
    all_weights = {}
    for name, module in train_model.named_modules():
        if hasattr(module, 'repvgg_convert'):
            kernel, bias = module.repvgg_convert()
            all_weights[name + '.rbr_reparam.weight'] = kernel
            all_weights[name + '.rbr_reparam.bias'] = bias
            print('convert RepVGG block')
        else:
            for p_name, p_tensor in module.named_parameters():
                full_name = name + '.' + p_name
                if full_name not in all_weights:
                    all_weights[full_name] = p_tensor.detach().cpu().numpy()
            for p_name, p_tensor in module.named_buffers():
                full_name = name + '.' + p_name
                if full_name not in all_weights:
                    all_weights[full_name] = p_tensor.cpu().numpy()

    deploy_model.load_state_dict(all_weights)
    if save_path is not None:
        torch.save(deploy_model.state_dict(), save_path)

    return deploy_model


#   Use this when converting a RepVGG without customized structures.
#   train_model = create_RepVGG_A0(deploy=False)
#   train train_model
#   deploy_model = repvgg_model_convert(train_model, create_RepVGG_A0, save_path='repvgg_deploy.pth')
def repvgg_model_convert(model: torch.nn.Module, build_func, save_path=None):
    converted_weights = {}
    scales = {}
    for name, module in model.named_modules():
        if hasattr(module, 'repvgg_convert'):
            kernel, bias = module.repvgg_convert()
            converted_weights[name + '.rbr_reparam.weight'] = kernel
            converted_weights[name + '.rbr_reparam.bias'] = bias
        elif isinstance(module, torch.nn.Linear):
            converted_weights[name + '.weight'] = module.weight.detach().cpu().numpy()
            converted_weights[name + '.bias'] = module.bias.detach().cpu().numpy()
    for name, params in model.named_parameters():
        if 'scale' in name:
            scales[name] = params
    del model

    deploy_model = build_func(deploy=True)
    for name, param in deploy_model.named_parameters():
        if 'scale' not in name:
            print('deploy param: ', name, param.size(), np.mean(converted_weights[name]))
            param.data = torch.from_numpy(converted_weights[name]).float()
        else:
            param.data = scales[name]

    if save_path is not None:
        torch.save(deploy_model.state_dict(), save_path)

    return deploy_model


def squeeze_weight(i, origin, indices, bias):
    if i == 0:
        select = torch.nonzero(indices[i]).squeeze()
        out = torch.index_select(origin, 0, select)
    else:
        if i != len(indices)-1:
            select1 = torch.nonzero(indices[i]).squeeze()
            out = torch.index_select(origin, 0, select1)
        else:
            out = origin
        if not bias:
            select2 = torch.nonzero(indices[i - 1]).squeeze()
            out = torch.index_select(out, 1, select2)
    return out


def create_model_from_checkpoint(architecture, checkpoint, deploy=False):
    # load checkpoint
    state_dict = torch.load(checkpoint, map_location='cpu')
    if 'state_dict' in state_dict:
        state_dict = state_dict['state_dict']
    ckpt = {k.replace('module.', ''): v for k, v in state_dict.items()}  # strip the names
    # build checkpoint model
    repvgg_build_func = get_RepVGG_func_by_name(architecture)
    model = repvgg_build_func(deploy=deploy)
    model.load_state_dict(ckpt)
    # convert model to deploy VGG shaped
    deploy_model = repvgg_model_convert(model, repvgg_build_func, save_path=None)

    # store weights waiting to be squeezed
    weights = {}
    for name, params in deploy_model.named_parameters():
        if 'reparam' in name or 'linear' in name:
            weights[name] = params

    # pull index from 'conv1x1' in origin checkpoint
    width = []
    indices = []
    for layer in ckpt:
        if 'scale' in layer:
            weight = ckpt[layer]
            binary = (weight > 0.5).float()
            width.append(torch.count_nonzero(binary).item())
            indices.append(binary.squeeze())

    # build final fine-shaped model
    final_model_build = get_RepVGG_func_by_name('RepVGG-config')
    final_model = final_model_build(deploy=True)

    # load parameters
    i = 0
    for name, params in final_model.named_parameters():
        if 'reparam' in name:
            squeezed = squeeze_weight(i, weights[name], indices, bias='bias' in name)
            params.data = squeezed
            print('processing: ', name, params.data.size())
            if 'bias' in name:
                i += 1
        if 'linear' in name:
            params.data = ckpt[name]
            print('processing: ', name, params.data.size())

    return model, final_model


if __name__ == '__main__':
    # state_dict = torch.load('B1_4_7487_9216.pth.tar', map_location='cpu')
    # if 'state_dict' in state_dict:
    #     state_dict = state_dict['state_dict']
    # for layer in state_dict:
    #     if 'scale' in layer:
    #         state_dict[layer] = (state_dict[layer] > 0.5).float()
    #         print(layer, 'remained: ', (torch.sum(state_dict[layer]) / (state_dict[layer]).size(0)).item())
    # torch.save(state_dict, 'B1_3G_convert_checkpoint.pt')

    # config = [[11], [29, 32, 27, 23], [68, 57, 55, 65, 79, 79],
    #           [208, 198, 193, 175, 201, 240, 254, 323, 357, 378, 412, 442, 483, 496, 510, 487], [2048]]
    # for i in config:
    #     print(i)
    #     print(len(i))
    # print(config[3])
    from prunesearch import ProfileConv
    from prunesearch import bn_prune

    x = torch.randn(1, 3, 224, 224)
    # indice = torch.tensor([1, 0, 1])
    # nz = torch.nonzero(indice).squeeze()
    # y = torch.index_select(x, 1, nz)
    # print(x, y)
    # repvgg_build_func = get_RepVGG_func_by_name('RepVGG-config')
    # model = repvgg_build_func(deploy=False)
    deploy, model = create_model_from_checkpoint('RepVGG-B1', 'checkpoint.pth.tar', )
    # print(model)
    deploy.eval()
    model.eval()
    out1 = deploy(x)
    out2 = model(x)
    print(torch.sum((out2 - out1) ** 2))

    # torch.onnx.export(model, x, "B1__G.onnx", verbose=True)
    # print('onnx exported')
    profile = ProfileConv(model)
    MACs = profile(torch.randn(1, 3, 224, 224))
    print(len(MACs))
    print(sum(MACs) / 1e9, 'GMACs, only consider conv layers')

    # model = create_model_from_checkpoint('RepVGG-B1', 'checkpoints/B1_4_7487_9216.pth.tar', )
    # # print(model)
    #
    # width = []
    # indices = []
    #
    # for name, params in model.named_parameters():
    #     if 'scale' in name:
    #         weight = params.detach()
    #         binary = (weight > 0.5).float()
    #         width.append(torch.count_nonzero(binary).item())
    #         indices.append(binary.squeeze())
    #
    # print(width)
    # print(indices)

    pass

    # profile = ProfileConv(model)
    # MACs = profile(torch.randn(1, 3, 224, 224))
    # print(len(MACs))
    # print(sum(MACs) / 1e9, 'GMACs, only consider conv layers')
