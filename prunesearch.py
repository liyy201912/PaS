import torch
import torch.nn as nn
import torchvision
import numpy as np
import math
from resnet import BinaryConv2d

# try:
#     from apex.parallel import DistributedDataParallel as DDP
#     from apex.fp16_utils import *
#     from apex import amp, optimizers
#     from apex.multi_tensor_apply import multi_tensor_applier
# except ImportError:
#     raise ImportError("Please install apex from https://www.github.com/nvidia/apex to run.")


resnet50_list = [118013952, 12845056, 115605504, 51380224, 51380224, 115605504, 51380224, 51380224, 115605504, 51380224,
                 102760448, 115605504, 51380224, 51380224, 115605504, 51380224, 51380224, 115605504, 51380224, 51380224,
                 115605504, 51380224, 102760448, 115605504, 51380224, 51380224, 115605504, 51380224, 51380224,
                 115605504,
                 51380224, 51380224, 115605504, 51380224, 51380224, 115605504, 51380224, 51380224, 115605504, 51380224,
                 102760448, 115605504, 51380224, 51380224, 115605504, 51380224, 51380224, 115605504, 51380224]

resnet50_mac_list = [614656.0, 3136.0, 28224.0, 3136.0, 3136.0, 28224.0, 3136.0, 3136.0, 28224.0, 3136.0, 3136.0,
                     7056.0, 784.0, 784.0, 7056.0, 784.0, 784.0, 7056.0, 784.0, 784.0, 7056.0, 784.0, 784.0, 1764.0,
                     196.0, 196.0, 1764.0, 196.0, 196.0, 1764.0, 196.0, 196.0, 1764.0, 196.0, 196.0, 1764.0, 196.0,
                     196.0, 1764.0, 196.0, 196.0, 441.0, 49.0, 49.0, 441.0, 49.0, 49.0, 441.0, 49.0]

resnet50_channel_list = [64, 64, 64, 256, 64, 64, 256, 64, 64, 256, 128, 128, 512, 128, 128, 512, 128, 128, 512,
                         128, 128, 512, 256, 256, 1024, 256, 256, 1024, 256, 256, 1024, 256, 256, 1024, 256,
                         256, 1024, 256, 256, 1024, 512, 512, 2048, 512, 512, 2048, 512, 512, 2048]

resnet50_in_list = [3, 64, 64, 64, 256, 64, 64, 256, 64, 64, 256, 128, 128, 512, 128, 128, 512, 128, 128, 512,
                    128, 128, 512, 256, 256, 1024, 256, 256, 1024, 256, 256, 1024, 256, 256, 1024, 256,
                    256, 1024, 256, 256, 1024, 512, 512, 2048, 512, 512, 2048, 512, 512]

repvgg_b1_list = [21676032.0, 231211008.0, 462422016.0, 462422016.0, 462422016.0, 231211008.0, 462422016.0,
                  462422016.0, 462422016.0, 462422016.0, 462422016.0, 231211008.0, 462422016.0, 462422016.0,
                  462422016.0, 462422016.0, 462422016.0, 462422016.0, 462422016.0, 462422016.0, 462422016.0,
                  462422016.0, 462422016.0, 462422016.0, 462422016.0, 462422016.0, 462422016.0, 462422016.0]

repvgg_b1_mac_list = [112896.0, 28224.0, 28224.0, 28224.0, 28224.0, 7056.0, 7056.0, 7056.0, 7056.0, 7056.0, 7056.0,
                      1764.0, 1764.0, 1764.0, 1764.0, 1764.0, 1764.0, 1764.0, 1764.0, 1764.0, 1764.0,
                      1764.0, 1764.0, 1764.0, 1764.0, 1764.0, 1764.0, 441.0]

repvgg_channel_list = [64, 128, 128, 128, 128, 256, 256, 256, 256, 256, 256,
                       512, 512, 512, 512, 512, 512, 512, 512, 512, 512, 512, 512, 512, 512, 512, 512, 2048]

mobilenet_list = [10838016.0, 3612672.0, 6422528.0, 19267584.0, 2709504.0, 7225344.0, 10838016.0, 4064256.0,
                  10838016.0, 10838016.0, 1016064.0, 3612672.0, 4816896.0, 1354752.0, 4816896.0, 4816896.0,
                  1354752.0, 4816896.0, 4816896.0, 338688.0, 2408448.0, 4816896.0, 677376.0, 4816896.0, 4816896.0,
                  677376.0, 4816896.0, 4816896.0, 677376.0, 4816896.0, 4816896.0, 677376.0, 7225344.0, 10838016.0,
                  1016064.0, 10838016.0, 10838016.0, 1016064.0, 10838016.0, 10838016.0, 254016.0, 4515840.0,
                  7526400.0, 423360.0, 7526400.0, 7526400.0, 423360.0, 7526400.0, 7526400.0, 423360.0,
                  15052800.0, 20070400.0]

mobilenet_mac_list = [112896.0, 3528.0, 12544.0, 12544.0, 294.0, 3136.0, 3136.0, 196.0, 3136.0, 3136.0, 49.0,
                      784.0, 784.0, 36.75, 784.0, 784.0, 36.75, 784.0, 784.0, 9.1875, 196.0, 196.0, 4.59375,
                      196.0, 196.0, 4.59375, 196.0, 196.0, 4.59375, 196.0, 196.0, 4.59375, 196.0, 196.0, 3.0625,
                      196.0, 196.0, 3.0625, 196.0, 196.0, 0.765625, 49.0, 49.0, 0.459375, 49.0, 49.0, 0.459375, 49.0,
                      49.0, 0.459375, 49.0, 49.0]

mobilenet_channel_list = [32, 32, 16, 96, 96, 24, 144, 144, 24, 144, 144, 32, 192, 192, 32, 192, 192, 32, 192, 192,
                          64, 384, 384, 64, 384, 384, 64, 384, 384, 64, 384, 384, 96, 576, 576, 96, 576, 576, 96,
                          576, 576, 160, 960, 960, 160, 960, 960, 160, 960, 960, 320, 1280]


class Distill(object):
    def __init__(self, student, teacher, optimizer, branch=True, rho=0.2):
        self.student = student
        self.teacher = teacher
        self.branch = branch
        self.rho = rho
        name_student = ['conv1', 'layer1', 'layer2', 'layer3', 'layer4']
        name_teacher = ['conv1', 'layer1', 'layer2', 'layer3', 'layer4']
        self.criterion = nn.MSELoss()
        self.branch_loss = 0.0
        if self.branch:
            self.branch_nets = []
            student_dim = [8, 8, 16, 32, 64]
            teacher_dim = [16, 16, 32, 64, 128]

            # for name, module in student.named_children():
            #     if name in name_student:
            #         student_dim.append(module.weight.size(0))
            # for name, module in teacher.named_children():
            #     if name in name_teacher:
            #         teacher_dim.append(module.weight.size(0))

            for idx, dim in enumerate(student_dim):
                self.branch_nets.append(nn.Sequential(nn.Conv2d(dim, teacher_dim[idx], kernel_size=1,
                                                                stride=1,
                                                                padding=0), ).to(next(student.parameters()).device))
            for net in self.branch_nets:
                optimizer.add_param_group({'params': net.parameters()})

        self.student_features = []
        self.teacher_features = []

        def hook_student(module, input, output):
            # print(module)
            self.student_features.append(output)

        def hook_teacher(module, input, output):
            # print(module)
            self.teacher_features.append(output)

        for name, module in self.student.named_children():
            if name in name_student:
                module.register_forward_hook(hook_student)
        for name, module in self.teacher.named_children():
            if name in name_teacher:
                module.register_forward_hook(hook_teacher)

    def train(self):
        for idx, feature in enumerate(self.student_features):
            # print(idx, len(self.teacher_features))
            # print(idx, len(self.student_features))
            if self.branch:
                branch_loss = self.rho * self.criterion(self.branch_nets[idx](feature), self.teacher_features[idx])
            else:
                branch_loss = self.rho * self.criterion(feature, self.teacher_features[idx])
            self.branch_loss += branch_loss.item()
            branch_loss.backward(retain_graph=True)

        print('distill loss: ', self.branch_loss)
        self.branch_loss = 0.0

    def clear(self):
        self.student_features.clear()
        self.teacher_features.clear()


# def append_loss(model, percent=0.5, alpha=5., mac_list=resnet50_list):
#     # a more automatic way?
#     alpha_adjust = alpha
#     # do the job
#     Branches = torch.tensor([]).cuda()
#     weights = torch.tensor([]).cuda()
#     # n = torch.tensor(0.).cuda()
#     macs = torch.tensor(mac_list).cuda() / 1e9
#     i = 0
#     for name, module in model.module.named_modules():
#         if 'scale' in name or isinstance(module, BinaryConv2d):
#             size = torch.squeeze(module.weight).size()
#             Branches = torch.cat((Branches, torch.squeeze(module.weight)), dim=0)
#             weights = torch.cat((weights, torch.ones(size, requires_grad=False).cuda() / size[0] * macs[i]), dim=0)
#             i += 1
#
#     # binarize
#     w = Branches.detach()
#     binary_w = (w > 0.5).float()
#     residual = w - binary_w
#     branch_out = Branches - residual
#     # compute regularization loss
#     total_count = torch.sum(branch_out * weights)
#     criterion = nn.MSELoss()
#     # branch_loss = criterion(total_count, n * math.sqrt(1 - percent))
#     branch_loss = criterion(total_count, torch.sum(macs) * math.sqrt(1 - percent))
#
#     return branch_loss * alpha_adjust


def append_loss(model, percent=0.66, alpha=5, arch='resnet50'):
    if arch == 'resnet50':
        alpha_adjust = alpha
        Branches = torch.tensor([]).cuda()
        for name, module in model.module.named_modules():
            if 'scale' in name or isinstance(module, BinaryConv2d):
                w = module.weight.detach()
                binary_w = (w > 0.5).float()
                residual = w - binary_w
                branch_out = module.weight - residual
                Branches = torch.cat((Branches, torch.sum(torch.squeeze(branch_out), dim=0, keepdim=True)), dim=0)

        target_macs = torch.tensor(sum(resnet50_list) / 1e9).cuda()
        in_channel = torch.cat((torch.tensor([3]).cuda(), Branches[:-1]), dim=0)
        # whether to use old
        # compare = torch.tensor(resnet50_in_list).cuda() - in_channel.detach()
        # for i in range(len(in_channel)):
        #     if i > 3 and (i - 4) % 3 == 0:
        #         in_channel[i] += compare[i]
        # print(in_channel)
        current_macs = torch.sum(torch.tensor(in_channel) * torch.tensor(resnet50_mac_list).cuda() * Branches) / 1e9
        criterion = nn.MSELoss()
        branch_loss = criterion(current_macs, target_macs * (1 - percent))
    elif arch == 'preresnet':
        alpha_adjust = alpha
        Branches = torch.tensor([]).cuda()
        for name, module in model.module.named_modules():
            if 'scale' in name or isinstance(module, BinaryConv2d):
                w = module.weight.detach()
                binary_w = (w > 0.5).float()
                residual = w - binary_w
                branch_out = module.weight - residual
                Branches = torch.cat((Branches, torch.sum(torch.squeeze(branch_out), dim=0, keepdim=True)), dim=0)

        target_macs = torch.tensor(sum(resnet50_list) / 1e9).cuda()
        out_channel = torch.cat((Branches, torch.tensor([2048.]).cuda()), dim=0)
        in_channel = torch.cat((torch.tensor([3]).cuda(), out_channel[:-1]), dim=0)
        # whether to use old
        # compare = torch.tensor(resnet50_in_list).cuda() - in_channel.detach()
        # for i in range(len(in_channel)):
        #     if i > 3 and (i - 4) % 3 == 0:
        #         in_channel[i] += compare[i]
        # print(in_channel)
        current_macs = torch.sum(torch.tensor(in_channel) * torch.tensor(resnet50_mac_list).cuda() * out_channel) / 1e9
        criterion = nn.MSELoss()
        branch_loss = criterion(current_macs, target_macs * (1 - percent))
    elif 'RepVGG' in arch:
        alpha_adjust = 0.1
        Branches = torch.tensor([]).cuda()
        # weights = torch.tensor([]).cuda()
        # n = torch.tensor(0.).cuda()
        for name, module in model.module.named_modules():
            if 'scale' in name or isinstance(module, BinaryConv2d):
                w = module.weight.detach()
                binary_w = (w > 0.5).float()
                residual = w - binary_w
                branch_out = module.weight - residual
                Branches = torch.cat((Branches, torch.sum(torch.squeeze(branch_out), dim=0, keepdim=True)), dim=0)
                # weights = torch.cat((weights, torch.ones(torch.squeeze(module.weight).size(),
                #                                          requires_grad=False).cuda() / torch.squeeze(module.weight).size(
                #     0)), dim=0)
                # weights = torch.cat((weights, torch.ones(torch.squeeze(module.weight).size(),
                #                                          requires_grad=False).cuda()), dim=0)
                # n += 1.

        # binarize
        # w = Branches.detach()
        # binary_w = (w > 0.5).float()
        # residual = w - binary_w
        # branch_out = Branches - residual
        # compute regularization loss
        # weights = weights / weights.size(0)
        # total_count = torch.sum(branch_out * weights)
        target_macs = torch.tensor(sum(repvgg_b1_list) / 1e9).cuda()
        in_channel = torch.cat((torch.tensor([3]).cuda(), Branches[:-1]), dim=0)
        current_macs = torch.sum(torch.tensor(in_channel) * torch.tensor(repvgg_b1_mac_list).cuda() * Branches) / 1e9
        criterion = nn.MSELoss()
        # branch_loss = criterion(total_count, n * math.sqrt(1 - percent))
        branch_loss = criterion(current_macs, target_macs * (1 - percent))
    elif arch == 'mobilenet':
        alpha_adjust = alpha
        Branches = torch.tensor([], requires_grad=True).cuda()
        for name, module in model.module.named_modules():
            if 'scale' in name or isinstance(module, BinaryConv2d):
                w = module.weight.detach()
                binary_w = (w > 0.5).float()
                residual = w - binary_w
                branch_out = module.weight - residual
                Branches = torch.cat((Branches, torch.sum(torch.squeeze(branch_out), dim=0, keepdim=True)), dim=0)

        out_channels = torch.cat((Branches, torch.tensor([1280], requires_grad=True).cuda()), dim=0)
        target_macs = torch.tensor(sum(resnet50_list) / 1e9).cuda()
        in_channel = torch.cat((torch.tensor([3], requires_grad=True).cuda(), out_channels[:-1]), dim=0)
        # whether to use old
        # compare = torch.tensor(resnet50_in_list).cuda() - in_channel.detach()
        # for i in range(len(in_channel)):
        #     if i > 3 and (i - 4) % 3 == 0:
        #         in_channel[i] += compare[i]
        # print(in_channel)
        current_macs = torch.sum(in_channel * torch.tensor(mobilenet_mac_list).cuda() * out_channels) / 1e9
        criterion = nn.MSELoss().cuda()
        branch_loss = criterion(current_macs, target_macs * (1 - percent))
    else:
        raise NotImplementedError

    return branch_loss * alpha_adjust


def pull_bn(model, percent):
    BNs = torch.tensor([]).cuda()
    weights = torch.tensor([]).cuda()
    length = []
    names = []
    n = 0.
    for name, module in model.module.named_modules():
        if 'bn' in name and 'downsample' not in name:
            BNs = torch.cat((BNs, module.weight.data), dim=0)
            weights = torch.cat((weights, torch.ones(module.weight.data.size()).cuda() / module.weight.data.size(0)),
                                dim=0)
            length.append(module.weight.data.size(0))
            names.append('module.' + name + '.weight')
            n += 1.

    srt, idx = torch.sort(torch.clone(BNs))
    sort_weights = weights[idx]
    accumulation = 0.
    j = 0.
    for i in range(sort_weights.size(0)):
        accumulation += sort_weights[i]
        j += 1.
        if accumulation > n * percent:
            break
    _, final_idx = torch.topk(BNs, int(j), largest=False)
    BNs[final_idx] = 0.

    sep = torch.split(BNs, length)
    output_percent = []
    for obj in sep:
        output_percent.append((1. - (torch.count_nonzero(obj) / obj.size(0))).item())

    return output_percent, names


def bn_prune(model, percent, arch='resnet'):
    if arch == 'resnet':
        BNs = torch.tensor([]).cuda()
        weights = torch.tensor([]).cuda()
        length = []
        # names = []
        n = 0
        for name, module in model.module.named_modules():
            if 'bn' in name and 'downsample' not in name:
                data = module.weight.data.detach()
                BNs = torch.cat((BNs, data), dim=0)
                weights = torch.cat((weights, torch.ones(data.size()).cuda() * n), dim=0)
                length.append(data.size(0))
                # names.append('module.' + name + '.weight')
                n += 1

        srt, idx = torch.sort(torch.clone(BNs))
        sort_weights = weights[idx]
        lis = resnet50_channel_list
        # accumulation = 0.
        j = 0
        for i in range(sort_weights.size(0)):
            # accumulation += sort_weights[i]
            # meter.update(sort_weights[i])
            lis[int(sort_weights[i])] -= 1
            in_channel = [3] + lis[:-1]
            compare = torch.tensor(resnet50_in_list) - torch.tensor(in_channel)
            for m in range(len(in_channel)):
                if m > 3 and (m - 4) % 3 == 0:
                    in_channel[m] += compare[m].item()
            j += 1
            # print(sum(meter.macs_list)/1e9, sum(repvgg_b1_list)/1e9)
            if torch.sum(torch.tensor(in_channel) * torch.tensor(resnet50_mac_list) * torch.tensor(lis)) < sum(
                    resnet50_list) * (1 - percent):
                break
        _, final_idx = torch.topk(BNs, int(j), largest=False)
        BNs[final_idx] = 0

        BNs = (BNs > 0).float()

        sep = torch.split(BNs, length)
        count_layer = 0
        for name, module in model.named_modules():
            if 'scale' in name:
                module.weight.data *= sep[count_layer].reshape(sep[count_layer].size(0), 1, 1, 1)
                count_layer += 1
    elif arch == 'repvgg':
        BNs1 = torch.tensor([]).cuda()
        BNs3 = torch.tensor([]).cuda()
        weights = torch.tensor([]).cuda()
        length = []
        names = []
        n = 0
        for name, module in model.named_modules():
            if 'rbr_dense.bn' in name:
                module_weight = module.weight.data
                numbers = module.weight.data.size(0)
                BNs3 = torch.cat((BNs3, module_weight), dim=0)
                # try:
                #     per_macs = (repvgg_b1_list[n] / numbers + repvgg_b1_list[n+1] / numbers) / 1e6
                # except:
                #     # per_macs = ((repvgg_b1_list[n] / numbers) / 1e6)
                #     per_macs = 0.
                # weights = torch.cat((weights, torch.ones(module.weight.data.size()).cuda() * per_macs), dim=0)
                weights = torch.cat((weights, torch.ones(module.weight.data.size()).cuda() * n), dim=0)

                length.append(numbers)
                names.append('module.' + name + '.weight')
                n += 1
            if 'rbr_1x1.bn' in name:
                module_weight = module.weight.data
                BNs1 = torch.cat((BNs1, module_weight), dim=0)

        BNs = BNs1 + BNs3

        srt, idx = torch.sort(torch.clone(BNs))
        sort_weights = weights[idx]

        # meter = MACsMeter(repvgg_b1_list, length)
        lis = repvgg_channel_list
        # accumulation = 0.
        j = 0
        for i in range(sort_weights.size(0)):
            # accumulation += sort_weights[i]
            # meter.update(sort_weights[i])
            lis[int(sort_weights[i])] -= 1
            in_channel = [3] + lis[:-1]
            j += 1
            # print(sum(meter.macs_list)/1e9, sum(repvgg_b1_list)/1e9)
            if torch.sum(torch.tensor(in_channel) * torch.tensor(repvgg_b1_mac_list) * torch.tensor(lis)) < sum(
                    repvgg_b1_list) * (1 - percent):
                break
        _, final_idx = torch.topk(BNs, j, largest=False)
        BNs[final_idx] = 0.

        binary_BNs = (torch.abs(BNs) > 1e-12).float()
        # print(binary_BNs)
        sep = torch.split(binary_BNs, length)
        # for obj in sep:
        #     print('binary bns', sum(obj) / len(obj))
        count_layer = 0
        for name, module in model.named_modules():
            if 'scale' in name:
                # print(module.weight.data)
                # print('processing ' + name,
                #       torch.sum(module.weight.data - sep[count_layer].reshape(sep[count_layer].size(0), 1, 1, 1)))
                # exit()
                # print(module.weight)
                module.weight.data = sep[count_layer].reshape(sep[count_layer].size(0), 1, 1, 1)
                # print(module.weight)
                # exit()
                # module.weight.data = module.weight.data
                count_layer += 1
        print('processed number of layers:', count_layer)
    else:
        raise NotImplementedError


class MACsMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, macs_list, width_list):
        self.macs_list = macs_list.copy()
        self.width_list = width_list.copy()

    def update(self, n):
        n = int(n)
        length = self.width_list[n]
        self.width_list[n] -= 1
        try:
            self.macs_list[n] -= self.macs_list[n] / length
            self.macs_list[n + 1] -= self.macs_list[n + 1] / length
        except:
            self.macs_list[n] -= self.macs_list[n] / length


def uniform_prune(model, percent):
    BNs = torch.tensor([]).cuda()
    weights = torch.tensor([]).cuda()
    length = []
    names = []
    n = 0.
    for name, module in model.module.named_modules():
        if 'bn' in name and 'downsample' not in name:
            BNs = torch.cat((BNs, module.weight.data), dim=0)
            weights = torch.cat((weights, torch.ones(module.weight.data.size()).cuda() / module.weight.data.size(0)),
                                dim=0)
            length.append(module.weight.data.size(0))
            names.append('module.' + name + '.weight')
            n += 1.

    srt, idx = torch.sort(torch.clone(BNs))
    sort_weights = weights[idx]
    accumulation = 0.
    j = 0.
    for i in range(sort_weights.size(0)):
        accumulation += sort_weights[i]
        j += 1.
        if accumulation > n * percent:
            break
    _, final_idx = torch.topk(BNs, int(j), largest=False)
    BNs[final_idx] = 0.

    sep = torch.split(BNs, length)
    output_percent = []
    for obj in sep:
        output_percent.append((1. - (torch.count_nonzero(obj) / obj.size(0))).item())

    return output_percent, names


class ProfileConv(nn.Module):
    def __init__(self, model):
        super(ProfileConv, self).__init__()
        self.model = model
        self.hooks = []
        self.macs = []

        def hook_conv(module, input, output):
            # if module.weight.size(-1) == 3:
            # if module.groups == 1:
            self.macs.append(output.size(1) * output.size(2) * output.size(3) *
                             module.weight.size(-1) * module.weight.size(-1) * input[0].size(1) / module.groups)
            # self.macs.append(output.size(2) * output.size(3) *
            #                  module.weight.size(-1) * module.weight.size(-1) / module.groups)
            # self.macs.append(output.size(1))

        for name, module in self.model.named_modules():
            # if isinstance(module, nn.Conv2d) and 'downsample' not in name:
            if isinstance(module, nn.Conv2d) and not isinstance(module, BinaryConv2d):
                print(name)
                self.hooks.append(module.register_forward_hook(hook_conv))

    def forward(self, x):
        self.model.to(x.device)
        _ = self.model(x)
        for handle in self.hooks:
            handle.remove()
        return self.macs


class Wrapmodel(nn.Module):
    def __init__(self, model):
        super(Wrapmodel, self).__init__()
        self.model = model
        self.hooks = []
        self.branch = {}
        for name, module in self.model.named_modules():
            if isinstance(module, nn.Conv2d):
                self.branch[name] = torch.ones(module.weight.size(), dtype=torch.float, requires_grad=True)

        def hook_conv(module, input, output):
            output *= 0.

        for name, module in self.model.named_modules():
            if isinstance(module, nn.Conv2d):
                self.hooks.append(module.register_forward_hook(hook_conv))

    def forward(self, x):
        # self.model.to(x.device)
        out = self.model(x)
        # for handle in self.hooks:
        #     handle.remove()
        return out


if __name__ == '__main__':
    # from resnet import BinaryConv2d
    #
    # model = BinaryConv2d(3, 3, 1, )
    # nn.init.uniform_(model.weight)
    # input = torch.randn(4, 3, 32, 32)
    # out = model(input)
    # a = torch.randn(2, 3, 4, 4)
    # b = torch.tensor([1, 0, 1])
    # print(a*b.reshape(1, -1, 1, 1))

    from newvgg import get_RepVGG_func_by_name
    from resnet import resnet50
    from autoprune import Convergence

    from mobilenet import mobilenet_v2

    # model = mobilenet_v2(pretrained=False)
    # print(model)
    # # for name, params in model.named_modules():
    # #     print(name)
    # profile = ProfileConv(model)
    # MACs = profile(torch.randn(1, 3, 224, 224))
    # print(MACs)
    # print(len(MACs))
    # print(sum(MACs) / 1e9, 'GMACs, only consider conv layers')
    # oc = torch.tensor(mobilenet_channel_list)
    # ic = torch.cat((torch.tensor([3]), oc[:-1]), dim=0)
    # print(oc, ic)
    # print(torch.sum(oc * ic * torch.tensor(mobilenet_mac_list)) / 1e9)
    # exit()

    # model = resnet50(pretrained=True)
    # # repvgg_build_func = get_RepVGG_func_by_name('RepVGG-B1')
    # # model = repvgg_build_func(deploy=False)
    #
    # # in_channel = torch.cat((torch.tensor([3]), torch.tensor(repvgg_channel_list)[:-1]), dim=0)
    # # print(in_channel)
    # # print(torch.sum(torch.tensor(in_channel)*torch.tensor(repvgg_b1_mac_list)*torch.tensor(repvgg_channel_list))/1e9)
    #
    # model = torch.nn.DataParallel(model.cuda())
    # # #
    # # state_dict = torch.load('checkpoints/B1_774.pth.tar', map_location='cpu')
    # # if 'state_dict' in state_dict:
    # #     state_dict = state_dict['state_dict']
    # # # ckpt = {k.replace('module.', ''): v for k, v in state_dict.items()}  # strip the names
    # #
    # # model.load_state_dict(state_dict, strict=False)
    # bn_prune(model, 0.65, arch='resnet')
    # print('current loss', append_loss(model, 0.65))
    # torch.save({'state_dict': model.state_dict()}, 'bn_pruned_b1.pt')
    # a = torch.randn(4, 1, 1, 1)
    # b = torch.tensor([1, 0, 1, 0])
    # print(a * (b.reshape(b.size(0), 1, 1, 1)))

    # for name, modules in model.named_modules():
    #     print(name)

    # model = torch.nn.DataParallel(resnet50(pretrained=True).cuda())

    # converge = Convergence(model)
    # converge.update(model)
    # converge.update(model)
    # converge.save()
    # checkpoint = torch.load('resnet_convergence.pt')
    # print(checkpoint)
    # torch.save(checkpoint, "resnet_convergence_16_1.pt", _use_new_zipfile_serialization=False)
    # exit()

    # in_channel = torch.cat((torch.tensor([3]), torch.tensor(resnet50_channel_list)[:-1]), dim=0)
    # print(torch.sum(torch.tensor(in_channel)*torch.tensor(resnet50_mac_list)*torch.tensor(resnet50_channel_list))/1e9)
    # bn_prune(model, 0.75, arch='resnet')
    # print(append_loss(model, 0.75, arch='resnet'))
    # model = torch.nn.DataParallel(resnet50(pretrained=True).cuda())
    # bn_prune(model, 0.75)
    # i = 0.
    # j = 0.
    # channel_list = []
    # for name, params in model.module.named_parameters():
    #     if 'scale' in name:
    #         print(name, 'remained: ', (torch.sum(params) / params.size(0)).item())
    #         channel_list.append(torch.sum(params).item())
    #         i += (torch.sum(params) / params.size(0)).item()
    #         j += 1.
    #
    # print('count ratio', i/j)
    # print(channel_list)
    # out_channel = channel_list
    # print(out_channel)
    # in_channel = [3] + out_channel[:-1]
    # # compare = torch.tensor(resnet50_in_list) - torch.tensor(in_channel)
    # # for i in range(len(in_channel)):
    # #     if i > 3 and (i - 4) % 3 == 0:
    # #         in_channel[i] += compare[i].item()
    # print(in_channel)
    # current_macs = torch.sum(
    #     torch.tensor(in_channel) * torch.tensor(resnet50_mac_list) * torch.tensor(out_channel)) / 1e9
    # print('pruned macs: ', current_macs)
    # exit()

    ################################################
    state_dict = torch.load('checkpoints/B1_4_7105.pth.tar', map_location='cpu')['state_dict']
    channel_list = []
    i = 0.
    j = 0.
    for layer in state_dict:
        if 'scale' in layer:
            state_dict[layer] = (state_dict[layer] > 0.5).float()
            print(layer, 'remained: ', (torch.sum(state_dict[layer]) / (state_dict[layer]).size(0)).item())
            i += (torch.sum(state_dict[layer]) / (state_dict[layer]).size(0)).item()
            j += 1.
            channel_list.append(torch.sum(state_dict[layer]).item())
            # channel_list.append(state_dict[layer].size(0))
    print('count ratio', i / j)
    out_channel = channel_list
    print(out_channel)
    in_channel = [3] + out_channel[:-1]
    # compare = torch.tensor(resnet50_in_list) - torch.tensor(in_channel)
    # for i in range(len(in_channel)):
    #     if i > 3 and (i - 4) % 3 == 0:
    #         in_channel[i] += compare[i].item()
    print(in_channel)
    current_macs = torch.sum(
        torch.tensor(in_channel) * torch.tensor(repvgg_b1_mac_list) * torch.tensor(out_channel)) / 1e9
    print('pruned macs: ', current_macs)
    # torch.save(state_dict, 'B1_3G_convert_checkpoint.pt')
    ###################################################

    # model = torchvision.models.resnet50(pretrained=False)
    # for name, module in model.named_modules():
    #     print(name, module.__name__)
    # percent, names = pull_bn(model, 0.75)
    # print(len(percent))
    # print(len(names))
    # print(w)
    # a = torch.tensor([1, 5, 3, 6])
    # b = torch.tensor([4, 5, 6, 7])
    # _, idx = torch.topk(a, 2, largest=False)
    # a[idx] = 0
    # print(a)
    # print(torch.ones(b.size()) / b.size(0))
    # for name, modules in model.named_modules():
    #     print(name)
    # loss = append_loss(model, 0.75)
    # print(loss.item())
    # a = torch.tensor([1, 2, 3])
    # b = torch.tensor([4, 5, 6])
    # c = a + b
    # c[2] = 0
    # print(a, b, c)
    # profile = ProfileConv(model)
    # MACs = profile(torch.randn(1, 3, 224, 224))
    # print(MACs)
    # print(len(MACs))
    # print(sum(MACs) / 1e9, 'GMACs, only consider conv layers')
