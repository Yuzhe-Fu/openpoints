# subsample layer for 3d processing.
from abc import ABC, abstractmethod

import torch
import torch.nn as nn
from torch.autograd import Function
import math
from openpoints.cpp.pointnet2_batch import pointnet2_cuda
import pdb
import asyncio
from concurrent.futures import ThreadPoolExecutor

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
        # pdb.set_trace()
    
    return torch.tensor(c)


# ################ make sure the total predict num is consistant
def adjust_list_to_sum(numbers, target_sum):
    current_sum = sum(numbers)
    if current_sum == target_sum:
        return numbers
    difference = target_sum - current_sum
    if(numbers[0] < numbers[1]):
        numbers[1] += difference
    else:
        numbers[0] += difference 
    return numbers


# recursive furthest point sampling by hancheng ye
def TreeBlock_fps_recursive_config(xyz, npoint, FPS_th, tree_depth=0, global_index=None, executor=None):
    if executor is None:
        # Create a single ThreadPoolExecutor to be shared
        with ThreadPoolExecutor(max_workers=64) as executor:
            return TreeBlock_fps_recursive_config(xyz, npoint, FPS_th, tree_depth, global_index, executor)

    if xyz.ndim == 3:
        B, N, _ = xyz.size()
    else:
        B = 1
        N = xyz.size(0)
        # xyz = xyz.unsqueeze(0)  # Add batch dimension if missing

    # Tree FPS: Partition and count
    if global_index is None:
        global_index = torch.arange(N).to(xyz.device).unsqueeze(0).repeat(B, 1)
    size_2dL_L1, xyz_2dL_L1, FPS_2dL_L1, global_index = part2_and_count_with_index(xyz, B, FPS_th, global_index, tree_depth)
    
    SampXyz = []
    index_global = []

    for i in range(B):  # Loop over batches
        PretNum_L1 = [0 for _ in range(2)]
        for j in range(2):
            PretNum_L1[j] = round(npoint * size_2dL_L1[i][j] / N)
        PretNum_L1_checked = adjust_list_to_sum(PretNum_L1, npoint)

        # Submit tasks to the shared executor
        futures = [executor.submit(process_branch, xyz_2dL_L1, size_2dL_L1, PretNum_L1_checked, FPS_2dL_L1, i, j, tree_depth + 1, global_index[i][j], FPS_th, executor) for j in range(2)]
        SampXyz_batch = [future.result()[0] for future in futures if future.result()[0] is not None]
        index_batch = [future.result()[1] for future in futures if future.result()[1] is not None]

        # Concatenate the results from all branches
        SampXyz_batch = torch.cat(SampXyz_batch, dim=1)
        index_batch = torch.cat(index_batch, dim=1)
        SampXyz.append(SampXyz_batch)
        index_global.append(index_batch)

    # Concatenate all batches
    SampXyz = torch.cat(SampXyz, dim=0)
    index_global = torch.cat(index_global, dim=0)
    return SampXyz, index_global

def part2_and_count_with_index(xyz, batch_size, FPS_th, ori_index, TreeDepth):
    # a list with (3, 4) shape
    # [max, min, mid, 1/4 point]
    direction = TreeDepth % 3

    max_val = torch.max(xyz[:, :, direction])
    min_val = torch.min(xyz[:, :, direction])
    mid_val = (max_val + min_val)/2

    xyz_0_temp = xyz[:,:,direction] < mid_val
    
    size_2dTensor  = [[0 for _ in range(2)] for _ in range(batch_size)]
    xyz_2dList = [[0 for _ in range(2)] for _ in range(batch_size)]
    FPS_2dList = [[0 for _ in range(2)] for _ in range(batch_size)]
    global_index = [[0 for _ in range(2)] for _ in range(batch_size)]

    for b in range(batch_size):
        for i in range(2):
            indx = torch.nonzero(xyz_0_temp[b,:]==i).reshape([1,-1]).unsqueeze(-1).expand(-1, -1, 3).to(torch.int64)
            xyz_row = (torch.gather(torch.reshape(xyz[b,:], [1,-1,3]), 1, indx))
            xyz_2dList[b][i] = xyz_row
            size_2dTensor[b][i] = indx.size()[1]
            global_index[b][i] = torch.gather(ori_index[b].reshape([1,-1]), 1, indx[..., 0])
            if indx.size()[1] <= FPS_th:
                FPS_2dList[b][i] = 1

    return size_2dTensor, xyz_2dList, FPS_2dList, global_index

def process_branch(xyz_2dL_L1, size_2dL_L1, PretNum_L1_checked, FPS_2dL_L1, i, j, tree_depth, global_index, FPS_th, executor=None):
    if size_2dL_L1[i][j] != 0:
        xyz_L1 = xyz_2dL_L1[i][j]
        SampNum_L1 = int(PretNum_L1_checked[j])
        if FPS_2dL_L1[i][j] == 1:
            indx_L1 = my_fps(xyz_L1, SampNum_L1)
            SampXyz_L1 = torch.gather(
                xyz_L1,
                1,
                indx_L1.unsqueeze(-1).long().expand(-1, -1, xyz_L1.shape[-1])
            ).type_as(xyz_L1)
            global_index_update = torch.gather(global_index, 1, indx_L1.long())
        else:
            SampXyz_L1, global_index_update = TreeBlock_fps_recursive_config(
                xyz_L1, SampNum_L1, FPS_th, tree_depth, global_index, executor
            )
        return SampXyz_L1, global_index_update
    else:
        return None, None




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


        if npoint==512:
            # output = TreeBlock_fps_depth10_config(xyz, npoint, 32).to('cuda:0')# actually spar 16 in algrithom 
            _, output = TreeBlock_fps_recursive_config(xyz, npoint, 32)
            # output = domain_fps_8block_sparto16(xyz, 8, npoint) # actually spar 16 in algrithom
            # output = fps_myown_seg(xyz, 16, npoint)
        elif npoint == 256:
            output = TreeBlock_fps_recursive_config(xyz, npoint, 32).to('cuda:0')# actually spar 16 in algrithom 
            # output = fps_myown_seg(xyz, 8, npoint)
        elif npoint == 128:
            output = TreeBlock_fps_recursive_config(xyz, npoint, 16).to('cuda:0')# actually spar 16 in algrithom 
            # output = fps_myown_seg(xyz, 4, npoint)
        elif npoint == 64:
            output = TreeBlock_fps_recursive_config(xyz, npoint, 16).to('cuda:0')# actually spar 16 in algrithom 
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
