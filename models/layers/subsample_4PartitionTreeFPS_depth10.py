# subsample layer for 3d processing.
from abc import ABC, abstractmethod

import torch
import torch.nn as nn
from torch.autograd import Function
import math
from openpoints.cpp.pointnet2_batch import pointnet2_cuda
import pdb

class BaseSampler(ABC):
    """If num_to_sample is provided, sample exactly
        num_to_sample points. Otherwise sample floor(pos[0] * ratio) points
    """

    def __init__(self, ratio=None, num_to_sample=None, subsampling_param=None):
        if num_to_sample is not None:
            if (ratio is not None) or (subsampling_param is not None):
                raise ValueError(
                    "Can only specify ratio or num_to_sample or subsampling_param, not several !")
            self._num_to_sample = num_to_sample

        elif ratio is not None:
            self._ratio = ratio

        elif subsampling_param is not None:
            self._subsampling_param = subsampling_param

        else:
            raise Exception(
                'At least ["ratio, num_to_sample, subsampling_param"] should be defined')

    def __call__(self, xyz):
        return self.sample(xyz)

    def _get_num_to_sample(self, npoints) -> int:
        if hasattr(self, "_num_to_sample"):
            return self._num_to_sample
        else:
            return math.floor(npoints * self._ratio)

    def _get_ratio_to_sample(self, batch_size) -> float:
        if hasattr(self, "_ratio"):
            return self._ratio
        else:
            return self._num_to_sample / float(batch_size)

    @abstractmethod
    def sample(self, xyz, feature=None, batch=None):
        pass


class RandomSample(BaseSampler):
    """Random Sample for dense data
        Arguments:
            xyz -- [B, N, 3]
    """

    def sample(self, xyz, **kwargs):
        if len(xyz.shape) != 3:
            raise ValueError(" Expects the xyz tensor to be of dimension 3")
        B, N, _ = xyz.shape
        idx = torch.randint(
            0, N, (B, self._get_num_to_sample(N)), device=xyz.device)
        sampled_xyz = torch.gather(xyz, 1, idx.unsqueeze(-1).expand(-1, -1, 3))
        # sampled_feature = torch.gather(feature, 2, idx.unsqueeze(1).repeat(1, C, 1))
        return sampled_xyz, idx


def my_fps(xyz, npoint):
    B, N, _ = xyz.size()
    output = torch.cuda.IntTensor(B, npoint)
    temp = torch.cuda.FloatTensor(B, N).fill_(1e10)
    pointnet2_cuda.furthest_point_sampling_wrapper(B, N, npoint, xyz, temp, output)
    return output

def random_sample(xyz, npoint):
    B, N, _ = xyz.shape
    idx = torch.randint(0, N, (B, npoint), device=xyz.device)
    return idx


def find_indices(a, b, batch):
    try:
        c = []
        for i in range(0, batch):
            c_matrix = []
            for b_row in b[i,:]:
                # 找到与b_row相等的行的索引
                a_row=a[i,:]
                idx = (a_row == b_row).all(dim=1).nonzero(as_tuple=True)[0]
                if idx.size()[0] == 1:
                    c_matrix.append(idx.item())
                else:
                    c_matrix.append(idx[0].item())
            c.append(c_matrix)
    except ValueError:
        print(ValueError)
        pdb.set_trace()
    
    return torch.tensor(c)


# ################ make sure the total predict num is consistant
def checkTotalNum(PredNumList, npoint):
    sumOfPN = sum(PredNumList)
    if sumOfPN != npoint:
        dif = npoint - sumOfPN
        max_value = max(PredNumList)
        max_position = PredNumList.index(max_value)
        PredNumList[max_position] = PredNumList[max_position] + dif
    return PredNumList


def adjust_list_to_sum(numbers, target_sum):
    # 计算列表的当前总和
    current_sum = sum(numbers)
    
    # 如果当前总和已经等于目标总和，则直接返回列表
    if current_sum == target_sum:
        return numbers

    # 计算需要调整的差值
    difference = target_sum - current_sum

    # 将列表中的元素按降序排序，并保留原始索引
    sorted_indices = sorted(range(len(numbers)), key=lambda i: numbers[i], reverse=True)

    # 从最大值开始，尝试逐步调整
    for i in range(len(sorted_indices)):
        index = sorted_indices[i]
        max_possible_adjustment = numbers[index] + difference
        if max_possible_adjustment >= 0:
            numbers[index] += difference
            return numbers
        else:
            # 如果减去差值后为负，则将负数的绝对值转移到下一个元素
            difference += numbers[index]
            numbers[index] = 0  # 当前元素设置为0，因为它无法承担更多的减少

    # 检查是否所有差值都已经被处理
    if difference == 0:
        return numbers
    else:
        raise ValueError("调整后的列表将包含负数，无法通过调整达到目标和。")


# ################ block partition and count the size
def block_partition_and_count(xyz, blockNum, batch_size, FPS_th):
    # a list with (3, 4) shape
    # [max, min, mid, 1/4 point]
    xyz_max_min_list = [[0 for _ in range(4)] for _ in range(3)]
    for i in range(3):
        max_val = torch.max(xyz[:, :, i])
        min_val = torch.min(xyz[:, :, i])
        mid_val = (max_val + min_val)/2
        fou_val = mid_val/2
        xyz_max_min_list[i][0] = max_val
        xyz_max_min_list[i][1] = min_val
        xyz_max_min_list[i][2] = (mid_val)
        xyz_max_min_list[i][3] = fou_val

    if blockNum == 2:
        xyz_0_temp = xyz[:,:,0] < xyz_max_min_list[0][2]
    elif blockNum == 4:
        xyz_0_temp = (xyz[:,:,0] < xyz_max_min_list[0][2]) + 2*(xyz[:,:,1] < xyz_max_min_list[1][2])
    elif blockNum == 8:
        xyz_0_temp = (xyz[:,:,0] < xyz_max_min_list[0][2]) + 2*(xyz[:,:,1] < xyz_max_min_list[1][2]) + 4*(xyz[:,:,2] < xyz_max_min_list[2][2])
    elif blockNum == 16:
        xyz_0_temp = (xyz[:,:,0] < xyz_max_min_list[0][2]) + 2*(xyz[:,:,1] < xyz_max_min_list[1][2]) + 4*(xyz[:,:,2] < xyz_max_min_list[2][2]) +8*(torch.abs(xyz[:,:,0]) < xyz_max_min_list[0][3])

    size_2dTensor  = [[0 for _ in range(blockNum)] for _ in range(batch_size)]
    xyz_2dList = [[0 for _ in range(blockNum)] for _ in range(batch_size)]
    FPS_2dList = [[0 for _ in range(blockNum)] for _ in range(batch_size)]

    
    for b in range(batch_size):
        for i in range(blockNum):
                indx = torch.nonzero(xyz_0_temp[b,:]==i).reshape([1,-1]).unsqueeze(-1).expand(-1, -1, 3).to(torch.int64)
                xyz_row = (torch.gather(torch.reshape(xyz[b,:], [1,-1,3]), 1, indx))
                xyz_2dList[b][i] = xyz_row
                size_2dTensor[b][i] = indx.size()[1]
                if indx.size()[1] <= FPS_th:
                    FPS_2dList[b][i] = 1

    return size_2dTensor, xyz_2dList, FPS_2dList

def TreeBlock_fps_depth10_config(xyz, npoint, blockNum, FPS_th):
    B, N, _ = xyz.size()
    # block-wise partition and predict -> can use small block Number to do this
    # PretNum_L1 = MultiStream_block_wise_Predict(xyz, npoint, blockNum[0], scale, PMS)
    # 如果二叉树，那起始无需预测，比如需采样512点，则第一层2叉树后每块采样256点

    # tree FPS
    # firstly tree block partition
    size_2dL_L1, xyz_2dL_L1, FPS_2dL_L1 = block_partition_and_count(xyz, blockNum[0], B, FPS_th)
    FPSlayer_list = []

    SampXyz = torch.tensor([], device='cuda')
    try:

        for i in range(0, B): # deep=1
            SampXyz_batch = torch.tensor([], device='cuda')
            PretNum_L1 = [0 for _ in range(blockNum[1])]
            for j in range(0, blockNum[0]):
                PretNum_L1[j] = round(npoint*size_2dL_L1[i][j]/N)
            PretNum_L1_checked = adjust_list_to_sum(PretNum_L1, npoint) # 确保累加起来是需要的值
            for j in range(0, blockNum[0]):
                if (size_2dL_L1[i][j] != 0):
                    xyz_L1 = xyz_2dL_L1[i][j]
                    SampNum_L1 = int(PretNum_L1_checked[j])
                    if (FPS_2dL_L1[i][j]==1):
                        indx_L1 = my_fps(xyz_L1, SampNum_L1)
                        SampXyz_L1 = torch.gather(xyz_L1, 1, indx_L1.unsqueeze(-1).long().expand(-1, -1, xyz_L1.shape[-1]))
                        SampXyz_batch = torch.cat((SampXyz_batch, SampXyz_L1), dim=1)
                    else: # 开始第二级树分块
                        size_2dL_L2, xyz_2dL_L2, FPS_2dL_L2 = block_partition_and_count(xyz_L1, blockNum[1], 1, FPS_th)
                        PretNum_L2 = [0 for _ in range(blockNum[1])]
                        for k in range(0, blockNum[1]):
                            PretNum_L2[k] = round(SampNum_L1*size_2dL_L2[0][k]/size_2dL_L1[i][j])
                        PretNum_L2_checked = adjust_list_to_sum(PretNum_L2, SampNum_L1) # 确保累加起来是需要的值
                        for k in range(0, blockNum[1]):
                            if (size_2dL_L2[0][k] != 0):
                                # check the size of FPS-2dL-L2
                                # pdb.set_trace()
                                xyz_L2 = xyz_2dL_L2[0][k]
                                SampNum_L2 = int(PretNum_L2_checked[k])
                                if (FPS_2dL_L2[0][k]==1):
                                    indx_L2_temp = my_fps(xyz_L2, SampNum_L2)
                                    SampXyz_L2 = torch.gather(xyz_L2, 1, indx_L2_temp.unsqueeze(-1).long().expand(-1, -1, xyz_L1.shape[-1]))
                                    SampXyz_batch = torch.cat((SampXyz_batch, SampXyz_L2), dim=1)
                                else: # 开始第三级树分块
                                    size_2dL_L3, xyz_2dL_L3, FPS_2dL_L3 = block_partition_and_count(xyz_L2, blockNum[2], 1, FPS_th) # actually the FPS_th2 is useless here
                                    # 这里要计算SampNum_L1在L3各个层上的加权平均数了
                                    PretNum_L3 = [0 for _ in range(blockNum[2])]
                                    for l in range(0, blockNum[2]):
                                        PretNum_L3[l] = round(SampNum_L2*size_2dL_L3[0][l]/size_2dL_L2[0][k])
                                    PretNum_L3_checked = adjust_list_to_sum(PretNum_L3, SampNum_L2) # 确保累加起来是需要的值
                                    for l in range(0, blockNum[2]):
                                        if (size_2dL_L3[0][l] != 0):
                                            xyz_L3 = xyz_2dL_L3[0][l]
                                            SampNum_L3 = int(PretNum_L3_checked[l])
                                            if (FPS_2dL_L3[0][l]==1):
                                                indx_L3_temp = my_fps(xyz_L3, SampNum_L3)
                                                SampXyz_L3 = torch.gather(xyz_L3, 1, indx_L3_temp.unsqueeze(-1).long().expand(-1, -1, xyz_L1.shape[-1]))
                                                SampXyz_batch = torch.cat((SampXyz_batch, SampXyz_L3), dim=1)
                                            else: # deep = 4
                                                size_2dL_L4, xyz_2dL_L4, FPS_2dL_L4 = block_partition_and_count(xyz_L3, blockNum[3], 1, FPS_th) # actually the FPS_th2 is useless here
                                                PretNum_L4 = [0 for _ in range(blockNum[3])]
                                                for m in range(0, blockNum[3]):
                                                    PretNum_L4[m] = round(SampNum_L3*size_2dL_L4[0][m]/size_2dL_L3[0][l])
                                                PretNum_L4_checked = adjust_list_to_sum(PretNum_L4, SampNum_L3) # 确保累加起来是需要的值
                                                for m in range(0, blockNum[3]):
                                                    if (size_2dL_L4[0][m] != 0):
                                                        xyz_L4 = xyz_2dL_L4[0][m]
                                                        SampNum_L4 = int(PretNum_L4_checked[m])
                                                        if (FPS_2dL_L4[0][m]==1):
                                                            indx_L4_temp = my_fps(xyz_L4, SampNum_L4)
                                                            SampXyz_L4 = torch.gather(xyz_L4, 1, indx_L4_temp.unsqueeze(-1).long().expand(-1, -1, xyz_L1.shape[-1]))
                                                            SampXyz_batch = torch.cat((SampXyz_batch, SampXyz_L4), dim=1)
                                                        else: #deep=5
                                                            size_2dL_L5, xyz_2dL_L5, FPS_2dL_L5 = block_partition_and_count(xyz_L4, blockNum[4], 1, FPS_th) # actually the FPS_th2 is useless here
                                                            PretNum_L5 = [0 for _ in range(blockNum[4])]
                                                            for n in range(0, blockNum[4]):
                                                                PretNum_L5[n] = round(SampNum_L4*size_2dL_L5[0][n]/size_2dL_L4[0][m])
                                                            PretNum_L5_checked = adjust_list_to_sum(PretNum_L5, SampNum_L4) # 确保累加起来是需要的值
                                                            for n in range(0, blockNum[4]):
                                                                if (size_2dL_L5[0][n] != 0):
                                                                    xyz_L5 = xyz_2dL_L5[0][n]
                                                                    SampNum_L5 = int(PretNum_L5_checked[n])
                                                                    if (FPS_2dL_L5[0][n]==1):
                                                                        indx_L5_temp = my_fps(xyz_L5, SampNum_L5)
                                                                        SampXyz_L5 = torch.gather(xyz_L5, 1, indx_L5_temp.unsqueeze(-1).long().expand(-1, -1, xyz_L1.shape[-1]))
                                                                        SampXyz_batch = torch.cat((SampXyz_batch, SampXyz_L5), dim=1)
                                                                    else:#deep 6
                                                                        size_2dL_L6, xyz_2dL_L6, FPS_2dL_L6 = block_partition_and_count(xyz_L5, blockNum[5], 1, FPS_th) # actually the FPS_th2 is useless here
                                                                        PretNum_L6 = [0 for _ in range(blockNum[5])]
                                                                        for o in range(0, blockNum[5]):
                                                                            PretNum_L6[o] = round(SampNum_L5*size_2dL_L6[0][o]/size_2dL_L5[0][n])
                                                                        PretNum_L6_checked = adjust_list_to_sum(PretNum_L6, SampNum_L5) # 确保累加起来是需要的值
                                                                        for o in range(0, blockNum[5]):
                                                                            if (size_2dL_L6[0][o] != 0):
                                                                                xyz_L6 = xyz_2dL_L6[0][o]
                                                                                SampNum_L6 = int(PretNum_L6_checked[o])
                                                                                if (FPS_2dL_L6[0][o]==1):
                                                                                    indx_L6_temp = my_fps(xyz_L6, SampNum_L6)
                                                                                    SampXyz_L6 = torch.gather(xyz_L6, 1, indx_L6_temp.unsqueeze(-1).long().expand(-1, -1, xyz_L1.shape[-1]))
                                                                                    SampXyz_batch = torch.cat((SampXyz_batch, SampXyz_L6), dim=1)
                                                                                else:#deep 7
                                                                                    size_2dL_L7, xyz_2dL_L7, FPS_2dL_L7 = block_partition_and_count(xyz_L6, blockNum[6], 1, FPS_th) # actually the FPS_th2 is useless here
                                                                                    PretNum_L7 = [0 for _ in range(blockNum[6])]
                                                                                    for p in range(0, blockNum[6]):
                                                                                        PretNum_L7[p] = round(SampNum_L6*size_2dL_L7[0][p]/size_2dL_L6[0][o])
                                                                                    PretNum_L7_checked = adjust_list_to_sum(PretNum_L7, SampNum_L6) # 确保累加起来是需要的值
                                                                                    for p in range(0, blockNum[6]):
                                                                                        if (size_2dL_L7[0][p] != 0):
                                                                                            xyz_L7 = xyz_2dL_L7[0][p]
                                                                                            SampNum_L7 = int(PretNum_L7_checked[p])
                                                                                            if (FPS_2dL_L7[0][p]==1):
                                                                                                indx_L7_temp = my_fps(xyz_L7, SampNum_L7)
                                                                                                SampXyz_L7 = torch.gather(xyz_L7, 1, indx_L7_temp.unsqueeze(-1).long().expand(-1, -1, xyz_L1.shape[-1]))
                                                                                                SampXyz_batch = torch.cat((SampXyz_batch, SampXyz_L7), dim=1)
                                                                                            else:#deep 8
                                                                                                size_2dL_L8, xyz_2dL_L8, FPS_2dL_L8 = block_partition_and_count(xyz_L7, blockNum[7], 1, FPS_th) # actually the FPS_th2 is useless here
                                                                                                PretNum_L8 = [0 for _ in range(blockNum[7])]
                                                                                                for q in range(0, blockNum[7]):
                                                                                                    PretNum_L8[q] = round(SampNum_L7*size_2dL_L8[0][q]/size_2dL_L7[0][p])
                                                                                                PretNum_L8_checked = adjust_list_to_sum(PretNum_L8, SampNum_L7) # 确保累加起来是需要的值
                                                                                                for q in range(0, blockNum[7]):
                                                                                                    if (size_2dL_L8[0][q] != 0):
                                                                                                        xyz_L8 = xyz_2dL_L8[0][q]
                                                                                                        SampNum_L8 = int(PretNum_L8_checked[q])
                                                                                                        if (FPS_2dL_L8[0][q]==1):
                                                                                                            indx_L8_temp = my_fps(xyz_L8, SampNum_L8)
                                                                                                            SampXyz_L8 = torch.gather(xyz_L8, 1, indx_L8_temp.unsqueeze(-1).long().expand(-1, -1, xyz_L1.shape[-1]))
                                                                                                            SampXyz_batch = torch.cat((SampXyz_batch, SampXyz_L8), dim=1)
                                                                                                        else:#deep 9
                                                                                                            size_2dL_L9, xyz_2dL_L9, FPS_2dL_L9 = block_partition_and_count(xyz_L8, blockNum[8], 1, FPS_th) # actually the FPS_th2 is useless here
                                                                                                            PretNum_L9 = [0 for _ in range(blockNum[8])]
                                                                                                            for r in range(0, blockNum[8]):
                                                                                                                PretNum_L9[r] = round(SampNum_L8*size_2dL_L9[0][r]/size_2dL_L8[0][q])
                                                                                                            PretNum_L9_checked = adjust_list_to_sum(PretNum_L9, SampNum_L8) # 确保累加起来是需要的值
                                                                                                            for r in range(0, blockNum[8]):
                                                                                                                if (size_2dL_L9[0][r] != 0):
                                                                                                                    xyz_L9 = xyz_2dL_L9[0][r]
                                                                                                                    SampNum_L9 = int(PretNum_L9_checked[r])
                                                                                                                    if (FPS_2dL_L9[0][r]==1):
                                                                                                                        indx_L9_temp = my_fps(xyz_L9, SampNum_L9)
                                                                                                                        SampXyz_L9 = torch.gather(xyz_L9, 1, indx_L9_temp.unsqueeze(-1).long().expand(-1, -1, xyz_L1.shape[-1]))
                                                                                                                        SampXyz_batch = torch.cat((SampXyz_batch, SampXyz_L9), dim=1)
                                                                                                                    else:#deep 10
                                                                                                                        size_2dL_L10, xyz_2dL_L10, FPS_2dL_L10 = block_partition_and_count(xyz_L9, blockNum[9], 1, FPS_th) # actually the FPS_th2 is useless here
                                                                                                                        PretNum_L10 = [0 for _ in range(blockNum[9])]
                                                                                                                        for s in range(0, blockNum[9]):
                                                                                                                            PretNum_L10[s] = round(SampNum_L9*size_2dL_L10[0][s]/size_2dL_L9[0][r])
                                                                                                                        PretNum_L10_checked = adjust_list_to_sum(PretNum_L10, SampNum_L9) # 确保累加起来是需要的值
                                                                                                                        for s in range(0, blockNum[9]):
                                                                                                                            if (size_2dL_L10[0][s] != 0):
                                                                                                                                xyz_L10 = xyz_2dL_L10[0][s]
                                                                                                                                SampNum_L10 = int(PretNum_L10_checked[s])
                                                                                                                                indx_L10_temp = my_fps(xyz_L10, SampNum_L10)
                                                                                                                                SampXyz_L10 = torch.gather(xyz_L10, 1, indx_L10_temp.unsqueeze(-1).long().expand(-1, -1, xyz_L1.shape[-1]))
                                                                                                                                SampXyz_batch = torch.cat((SampXyz_batch, SampXyz_L10), dim=1)



                            else:
                                continue
                else:
                    continue
            # pdb.set_trace()
            SampXyz = torch.cat((SampXyz, SampXyz_batch), dim=0)
                
        SampXyz_indx = find_indices(xyz, SampXyz, B)
    except RuntimeError:
            print(RuntimeError)
            pdb.set_trace()

    return SampXyz_indx


class FurthestPointSampling(Function):
    counter = 0
    @staticmethod
    def forward(ctx, xyz: torch.Tensor, npoint: int) -> torch.Tensor:
        """
        Uses iterative furthest point sampling to select a set of npoint features that have the largest
        minimum distance
        :param ctx:
        :param xyz: (B, N, 3) where N > npoint
        :param npoint: int, number of features in the sampled set
        :return:
             output: (B, npoint) tensor containing the set (idx)
        """
        # pdb.set_trace()
        assert xyz.is_contiguous()

        B, N, _ = xyz.size()
        # xyz = quanti32(xyz) # my function, convert xyz into 0~256
        # output = torch.cuda.IntTensor(B, npoint, device=xyz.device)
        # temp = torch.cuda.FloatTensor(B, N, device=xyz.device).fill_(1e10)
        # if npoint==512:
        #     output = fps_myown(xyz, npoint)
        # else:
        #     output = torch.cuda.IntTensor(B, npoint)
        #     temp = torch.cuda.FloatTensor(B, N).fill_(1e10)
        #     pointnet2_cuda.furthest_point_sampling_wrapper(B, N, npoint, xyz, temp, output)

        Blk = 2
        blockNum = [Blk,Blk,Blk,Blk,Blk,Blk,Blk,Blk,Blk,Blk] 
        FPS_th = 64

        if npoint==512:
            output = TreeBlock_fps_depth10_config(xyz, npoint, blockNum, 64).to('cuda:0')# actually spar 16 in algrithom 
            # output = domain_fps_8block_sparto16(xyz, 8, npoint) # actually spar 16 in algrithom
            # output = fps_myown_seg(xyz, 16, npoint)
        elif npoint == 256:
            output = TreeBlock_fps_depth10_config(xyz, npoint, blockNum, 64).to('cuda:0')# actually spar 16 in algrithom 
            # output = fps_myown_seg(xyz, 8, npoint)
        elif npoint == 128:
            output = TreeBlock_fps_depth10_config(xyz, npoint, blockNum, 32).to('cuda:0')# actually spar 16 in algrithom 
            # output = fps_myown_seg(xyz, 4, npoint)
        elif npoint == 64:
            output = TreeBlock_fps_depth10_config(xyz, npoint, blockNum, 32).to('cuda:0')# actually spar 16 in algrithom 
            # output = fps_myown_seg(xyz, 2, npoint)
        else:
            output = torch.cuda.IntTensor(B, npoint)
            temp = torch.cuda.FloatTensor(B, N).fill_(1e10)
            pointnet2_cuda.furthest_point_sampling_wrapper(B, N, npoint, xyz, temp, output)
        return output

    @staticmethod
    def backward(xyz, a=None):
        return None, None


furthest_point_sample = FurthestPointSampling.apply


class GatherOperation(Function):
    @staticmethod
    def forward(ctx, features: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
        """
        :param ctx:
        :param features: (B, C, N)
        :param idx: (B, npoint) index tensor of the features to gather
        :return:
            output: (B, C, npoint)
        """
        assert features.is_contiguous()
        assert idx.is_contiguous()

        B, npoint = idx.size()
        _, C, N = features.size()
        output = torch.cuda.FloatTensor(B, C, npoint, device=features.device)

        pointnet2_cuda.gather_points_wrapper(
            B, C, N, npoint, features, idx, output)

        ctx.for_backwards = (idx, C, N)
        return output

    @staticmethod
    def backward(ctx, grad_out):
        idx, C, N = ctx.for_backwards
        B, npoint = idx.size()

        grad_features = torch.zeros(
            [B, C, N], dtype=torch.float, device=grad_out.device, requires_grad=True)
        grad_out_data = grad_out.data.contiguous()
        pointnet2_cuda.gather_points_grad_wrapper(
            B, C, N, npoint, grad_out_data, idx, grad_features.data)
        return grad_features, None


gather_operation = GatherOperation.apply
# mark: torch gather is even faster. sampled_xyz = torch.gather(points, 1, idx.unsqueeze(-1).expand(-1, -1, 3))


def fps(data, number):
    '''
        data B N C
        number int
    '''
    fps_idx = furthest_point_sample(data[:, :, :3].contiguous(), number)
    fps_data = torch.gather(
        data, 1, fps_idx.unsqueeze(-1).long().expand(-1, -1, data.shape[-1]))
    return fps_data


if __name__ == '__main__':
    import time

    B, C, N = 2, 3, 10000
    K = 16
    device = 'cuda'
    points = torch.randn([B, N, 3], device=device, dtype=torch.float)
    print(points.shape, '\n', points)

    nsample = 4096
    idx = furthest_point_sample(points, nsample)

    st = time.time()
    for _ in range(100):
        query1 = torch.gather(
            points, 1, idx.long().unsqueeze(-1).expand(-1, -1, 3))
    print(time.time() - st)
    print(query1.shape)

    st = time.time()
    for _ in range(100):
        query2 = gather_operation(points.transpose(
            1, 2).contiguous(), idx).transpose(1, 2).contiguous()
    print(time.time() - st)
    print(query2.shape)

    print(torch.allclose(query1, query2))
