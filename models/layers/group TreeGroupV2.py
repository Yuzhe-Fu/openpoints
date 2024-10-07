# group layer: find neighbors for each point
# knn, knn_sparse, ball query

# gather layer, gather features by index
from typing import Tuple
import copy, logging
import torch
import torch.nn as nn
from torch.autograd import Function
from openpoints.cpp import pointnet2_cuda
import pdb

class KNN(nn.Module):
    def __init__(self, neighbors, transpose_mode=True):
        super(KNN, self).__init__()
        self.neighbors = neighbors

    @torch.no_grad()
    def forward(self, support, query):
        """
        Args:
            support ([tensor]): [B, N, C]
            query ([tensor]): [B, M, C]
        Returns:
            [int]: neighbor idx. [B, M, K]
        """
        dist = torch.cdist(support, query)
        k_dist = dist.topk(k=self.neighbors, dim=1, largest=False)
        return k_dist.values, k_dist.indices.transpose(1, 2).contiguous().int()

# dilated knn
class DenseDilated(nn.Module):
    """
    Find dilated neighbor from neighbor list
    index: (B, npoint, nsample)
    """

    def __init__(self, k=9, dilation=1, stochastic=False, epsilon=0.0):
        super(DenseDilated, self).__init__()
        self.dilation = dilation
        self.stochastic = stochastic
        self.epsilon = epsilon
        self.k = k

    def forward(self, edge_index):
        if self.stochastic:
            if torch.rand(1) < self.epsilon and self.training:
                num = self.k * self.dilation
                randnum = torch.randperm(num)[:self.k]
                edge_index = edge_index[:, :, randnum]
            else:
                edge_index = edge_index[:, :, ::self.dilation]
        else:
            edge_index = edge_index[:, :, ::self.dilation]
        return edge_index.contiguous()


class DilatedKNN(nn.Module):
    """
    Find the neighbors' indices based on dilated knn
    """

    def __init__(self, k=9, dilation=1, stochastic=False, epsilon=0.0):
        super(DilatedKNN, self).__init__()
        self.dilation = dilation
        self.stochastic = stochastic
        self.epsilon = epsilon
        self.k = k
        self._dilated = DenseDilated(k, dilation, stochastic, epsilon)
        self.knn = KNN(k * self.dilation, transpose_mode=True)

    def forward(self, query):
        _, idx = self.knn(query, query)
        return self._dilated(idx)


class GroupingOperation(Function):

    @staticmethod
    @torch.cuda.amp.custom_fwd(cast_inputs=torch.float32)
    def forward(ctx, features: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
        """
        :param ctx:
        :param features: (B, C, N) tensor of features to group
        :param idx: (B, npoint, nsample) tensor containing the indicies of features to group with
        :return:
            output: (B, C, npoint, nsample) tensor
        """
        assert features.is_contiguous()
        assert idx.is_contiguous()

        B, nfeatures, nsample = idx.size()
        _, C, N = features.size()
        output = torch.cuda.FloatTensor(B, C, nfeatures, nsample, device=features.device)

        pointnet2_cuda.group_points_wrapper(B, C, N, nfeatures, nsample, features, idx, output)

        ctx.for_backwards = (idx, N)
        return output

    @staticmethod
    def backward(ctx, grad_out: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        :param ctx:
        :param grad_out: (B, C, npoint, nsample) tensor of the gradients of the output from forward
        :return:
            grad_features: (B, C, N) gradient of the features
        """
        idx, N = ctx.for_backwards

        B, C, npoint, nsample = grad_out.size()
        grad_features = torch.zeros([B, C, N], dtype=torch.float, device=grad_out.device, requires_grad=True)
        grad_out_data = grad_out.data.contiguous()
        pointnet2_cuda.group_points_grad_wrapper(B, C, N, npoint, nsample, grad_out_data, idx, grad_features.data)
        return grad_features, None


grouping_operation = GroupingOperation.apply


def torch_grouping_operation(features, idx):
    r"""from torch points kernels
    Parameters
    ----------
    features : torch.Tensor
        (B, C, N) tensor of features to group
    idx : torch.Tensor
        (B, npoint, nsample) tensor containing the indicies of features to group with

    Returns
    -------
    torch.Tensor
        (B, C, npoint, nsample) tensor
    """
    all_idx = idx.reshape(idx.shape[0], -1)
    all_idx = all_idx.unsqueeze(1).repeat(1, features.shape[1], 1)
    grouped_features = features.gather(2, all_idx)
    return grouped_features.reshape(idx.shape[0], features.shape[1], idx.shape[1], idx.shape[2])


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

        pointnet2_cuda.gather_points_wrapper(B, C, N, npoint, features, idx, output)

        ctx.for_backwards = (idx, C, N)
        return output

    @staticmethod
    def backward(ctx, grad_out):
        idx, C, N = ctx.for_backwards
        B, npoint = idx.size()

        grad_features = torch.zeros([B, C, N], dtype=torch.float, device=grad_out.device, requires_grad=True)
        grad_out_data = grad_out.data.contiguous()
        pointnet2_cuda.gather_points_grad_wrapper(B, C, N, npoint, grad_out_data, idx, grad_features.data)
        return grad_features, None


gather_operation = GatherOperation.apply

# add TreeGroup here
# ################ block partition with different directions and count the size
def part2_and_count(xyz, FPS_th, TreeDepth):
    # a list with (3, 4) shape
    # [max, min, mid, 1/4 point]
    direction = TreeDepth % 3
    # xyz.size(), torch.Size([1, 1024, 3])
    xyz_onedirec = xyz[:, :, direction]
    xyz_not5 = xyz_onedirec[xyz_onedirec != 5.0]
    max_val = torch.max(xyz_not5)
    min_val = torch.min(xyz_not5)
    mid_val = (max_val + min_val)/2

    # double check the direction, incase the mid_val is equal to max or min
    if mid_val == max_val:
        xyz_onedirec = xyz[:, :, ((TreeDepth+1) % 3)]
        xyz_not5 = xyz_onedirec[xyz_onedirec != 5.0]
        max_val = torch.max(xyz_not5)
        min_val = torch.min(xyz_not5)
        mid_val = (max_val + min_val)/2
        assert max_val != min_val

    xyz_lessMid_indx = torch.nonzero((xyz_onedirec < mid_val))
    xyz_largMid_indx = torch.nonzero((xyz_onedirec >= mid_val) & (xyz_onedirec != 5))
    
    size_2dTensor  = [0 for _ in range(2)]
    xyz_2dList = [0 for _ in range(2)]
    FPS_2dList = [0 for _ in range(2)]

    size_2dTensor[0] = xyz_lessMid_indx.size()[0]
    size_2dTensor[1] = xyz_largMid_indx.size()[0]
    FPS_2dList[0] = 1 if size_2dTensor[0] <= FPS_th else 0
    FPS_2dList[1] = 1 if size_2dTensor[1] <= FPS_th else 0
    # pdb.set_trace()
    xyz_lessMid = torch.full_like(xyz, 5)
    xyz_lessMid[:, xyz_lessMid_indx[:,1], :] = xyz[:, xyz_lessMid_indx[:,1], :]
    xyz_largMid = torch.full_like(xyz, 5)
    xyz_largMid[:, xyz_largMid_indx[:,1], :] = xyz[:, xyz_largMid_indx[:,1], :]
    xyz_2dList[0] = xyz_lessMid
    xyz_2dList[1] = xyz_largMid

    return size_2dTensor, xyz_2dList, FPS_2dList

def selectAfromB(A, B):
    A_2d = A.squeeze(0)
    B_2d = B.squeeze(0)
    # pdb.set_trace()
    matches = (B_2d[:, None] == A_2d).all(-1).any(1)
    C = B_2d[matches]
    count = C.shape[0]
    return C.unsqueeze(0) , count


def mygroup(ctx, radius: float, nsample: int, xyz: torch.Tensor, new_xyz: torch.Tensor) -> torch.Tensor:
    """
    :param ctx:
    :param radius: float, radius of the balls
    :param nsample: int, maximum number of features in the balls
    :param xyz: (B, N, 3) xyz coordinates of the features
    :param new_xyz: (B, npoint, 3) centers of the ball query
    :return:
        idx: (B, npoint, nsample) tensor with the indicies of the features that form the query balls
    """
    assert new_xyz.is_contiguous()
    assert xyz.is_contiguous()

    B, N, _ = xyz.size()
    npoint = new_xyz.size(1)
    idx = torch.cuda.IntTensor(B, npoint, nsample, device=xyz.device).zero_()
    pointnet2_cuda.ball_query_wrapper(B, N, npoint, radius, nsample, new_xyz, xyz, idx)
    # pdb.set_trace()
    return idx


def Tree_group_depth10_config(xyz, nsample, new_xyz, radius, FPS_th):
    B, N, _ = xyz.size()

    # tree FPS
    # firstly tree block partition

    indx_all = torch.tensor([], device='cuda')
    try:
        for i in range(0, B): # deep=1
            indx_batch = torch.tensor([], device='cuda')
            size_2dL_L1, xyz_2dL_L1, FPS_2dL_L1 = part2_and_count(xyz[i, :, :].unsqueeze(0),FPS_th, 0)
            for j in range(0, 2):
                if (size_2dL_L1[j] != 0):
                    xyz_L1 = xyz_2dL_L1[j]
                    if (FPS_2dL_L1[j]==1):
                        centerXYZ, centerXYZ_size = selectAfromB(xyz_L1, new_xyz[i,:,:].unsqueeze(0))
                        if(centerXYZ_size != 0):
                            group_from_xyz = xyz_L1
                            group_from_xyz_N = xyz_L1.size(1)
                            idx = torch.cuda.IntTensor(1, centerXYZ_size, nsample, device=xyz.device).zero_()
                            pointnet2_cuda.ball_query_wrapper(1, group_from_xyz_N, centerXYZ_size, radius, nsample, centerXYZ, group_from_xyz, idx)
                            indx_batch = torch.cat((indx_batch, idx), dim=1)
                        else:
                            continue
                    else: # deep = 2
                        size_2dL_L2, xyz_2dL_L2, FPS_2dL_L2 = part2_and_count(xyz_L1, FPS_th, 1)
                        for k in range(0, 2):
                            if (size_2dL_L2[k] != 0):
                                xyz_L2 = xyz_2dL_L2[k]
                                if (FPS_2dL_L2[k]==1):
                                    centerXYZ, centerXYZ_size = selectAfromB(xyz_L2, new_xyz[i,:,:].unsqueeze(0))
                                    if(centerXYZ_size != 0):
                                        group_from_xyz = xyz_L1 # group source
                                        group_from_xyz_N = group_from_xyz.size(1)
                                        idx = torch.cuda.IntTensor(1, centerXYZ_size, nsample, device=xyz.device).zero_()
                                        pointnet2_cuda.ball_query_wrapper(1, group_from_xyz_N, centerXYZ_size, radius, nsample, centerXYZ, group_from_xyz, idx)
                                        indx_batch = torch.cat((indx_batch, idx), dim=1)
                                    else:
                                        continue
                                else: # deep = 3
                                    size_2dL_L3, xyz_2dL_L3, FPS_2dL_L3 = part2_and_count(xyz_L2, FPS_th, 2) # actually the FPS_th2 is useless here
                                    for l in range(0, 2):
                                        if (size_2dL_L3[l] != 0):
                                            xyz_L3 = xyz_2dL_L3[l]
                                            if (FPS_2dL_L3[l]==1):
                                                centerXYZ, centerXYZ_size = selectAfromB(xyz_L3, new_xyz[i,:,:].unsqueeze(0))
                                                if(centerXYZ_size != 0):
                                                    group_from_xyz = xyz_L2 # group source
                                                    group_from_xyz_N = group_from_xyz.size(1)
                                                    idx = torch.cuda.IntTensor(1, centerXYZ_size, nsample, device=xyz.device).zero_()
                                                    pointnet2_cuda.ball_query_wrapper(1, group_from_xyz_N, centerXYZ_size, radius, nsample, centerXYZ, group_from_xyz, idx)
                                                    indx_batch = torch.cat((indx_batch, idx), dim=1)
                                                else:
                                                    continue
                                            else: # deep = 4
                                                size_2dL_L4, xyz_2dL_L4, FPS_2dL_L4 = part2_and_count(xyz_L3, FPS_th, 3) # actually the FPS_th2 is useless here
                                                for m in range(0, 2):
                                                    if (size_2dL_L4[m]!=0):
                                                        xyz_L4 = xyz_2dL_L4[m]
                                                        if (FPS_2dL_L4[m]==1):
                                                            centerXYZ, centerXYZ_size = selectAfromB(xyz_L4, new_xyz[i,:,:].unsqueeze(0)) # group source
                                                            if(centerXYZ_size != 0):
                                                                group_from_xyz = xyz_L3 # group source
                                                                group_from_xyz_N = group_from_xyz.size(1)
                                                                idx = torch.cuda.IntTensor(1, centerXYZ_size, nsample, device=xyz.device).zero_()
                                                                pointnet2_cuda.ball_query_wrapper(1, group_from_xyz_N, centerXYZ_size, radius, nsample, centerXYZ, group_from_xyz, idx)
                                                                indx_batch = torch.cat((indx_batch, idx), dim=1)
                                                            else:
                                                                continue
                                                        else: #deep=5
                                                            size_2dL_L5, xyz_2dL_L5, FPS_2dL_L5 = part2_and_count(xyz_L4, FPS_th, 4) # actually the FPS_th2 is useless here
                                                            for n in range(0, 2):
                                                                if (size_2dL_L5[n]!=0):
                                                                    xyz_L5 = xyz_2dL_L5[n]
                                                                    if (FPS_2dL_L5[n]==1):
                                                                        centerXYZ, centerXYZ_size = selectAfromB(xyz_L5, new_xyz[i,:,:].unsqueeze(0)) # group source
                                                                        if(centerXYZ_size != 0):
                                                                            group_from_xyz = xyz_L4 # group source
                                                                            group_from_xyz_N = group_from_xyz.size(1)
                                                                            idx = torch.cuda.IntTensor(1, centerXYZ_size, nsample, device=xyz.device).zero_()
                                                                            pointnet2_cuda.ball_query_wrapper(1, group_from_xyz_N, centerXYZ_size, radius, nsample, centerXYZ, group_from_xyz, idx)
                                                                            indx_batch = torch.cat((indx_batch, idx), dim=1)
                                                                        else:
                                                                            continue
                                                                    else:#deep 6
                                                                        size_2dL_L6, xyz_2dL_L6, FPS_2dL_L6 = part2_and_count(xyz_L5, FPS_th, 5) # actually the FPS_th2 is useless here
                                                                        for o in range(0, 2):
                                                                            if (size_2dL_L6[o] != 0):
                                                                                xyz_L6 = xyz_2dL_L6[o]
                                                                                if (FPS_2dL_L6[o]==1):
                                                                                    centerXYZ, centerXYZ_size = selectAfromB(xyz_L6, new_xyz[i,:,:].unsqueeze(0)) # group source
                                                                                    if(centerXYZ_size != 0):
                                                                                        group_from_xyz = xyz_L5 # group source
                                                                                        group_from_xyz_N = group_from_xyz.size(1)
                                                                                        idx = torch.cuda.IntTensor(1, centerXYZ_size, nsample, device=xyz.device).zero_()
                                                                                        pointnet2_cuda.ball_query_wrapper(1, group_from_xyz_N, centerXYZ_size, radius, nsample, centerXYZ, group_from_xyz, idx)
                                                                                        indx_batch = torch.cat((indx_batch, idx), dim=1)
                                                                                    else:
                                                                                        continue
                                                                                else:#deep 7
                                                                                    size_2dL_L7, xyz_2dL_L7, FPS_2dL_L7 = part2_and_count(xyz_L6,FPS_th,6) # actually the FPS_th2 is useless here
                                                                                    for p in range(0, 2):
                                                                                        if (size_2dL_L7[p] != 0):
                                                                                            xyz_L7 = xyz_2dL_L7[p]
                                                                                            if (FPS_2dL_L7[p]==1):
                                                                                                centerXYZ, centerXYZ_size = selectAfromB(xyz_L7, new_xyz[i,:,:].unsqueeze(0)) # group source
                                                                                                if(centerXYZ_size != 0):
                                                                                                    group_from_xyz = xyz_L6 # group source
                                                                                                    group_from_xyz_N = group_from_xyz.size(1)
                                                                                                    idx = torch.cuda.IntTensor(1, centerXYZ_size, nsample, device=xyz.device).zero_()
                                                                                                    pointnet2_cuda.ball_query_wrapper(1, group_from_xyz_N, centerXYZ_size, radius, nsample, centerXYZ, group_from_xyz, idx)
                                                                                                    indx_batch = torch.cat((indx_batch, idx), dim=1)
                                                                                                else:
                                                                                                    continue
                                                                                            else:#deep 8
                                                                                                size_2dL_L8, xyz_2dL_L8, FPS_2dL_L8 = part2_and_count(xyz_L7,FPS_th,7) # actually the FPS_th2 is useless here
                                                                                                for q in range(0, 2):
                                                                                                    if (size_2dL_L8[q] != 0):
                                                                                                        xyz_L8 = xyz_2dL_L8[q]
                                                                                                        if (FPS_2dL_L8[q]==1):
                                                                                                            centerXYZ, centerXYZ_size = selectAfromB(xyz_L8, new_xyz[i,:,:].unsqueeze(0)) # group source
                                                                                                            if(centerXYZ_size != 0):
                                                                                                                group_from_xyz = xyz_L7 # group source
                                                                                                                group_from_xyz_N = group_from_xyz.size(1)
                                                                                                                idx = torch.cuda.IntTensor(1, centerXYZ_size, nsample, device=xyz.device).zero_()
                                                                                                                pointnet2_cuda.ball_query_wrapper(1, group_from_xyz_N, centerXYZ_size, radius, nsample, centerXYZ, group_from_xyz, idx)
                                                                                                                indx_batch = torch.cat((indx_batch, idx), dim=1)
                                                                                                            else:
                                                                                                                continue
                                                                                                        else:#deep 9
                                                                                                            size_2dL_L9, xyz_2dL_L9, FPS_2dL_L9 = part2_and_count(xyz_L8, FPS_th,8) # actually the FPS_th2 is useless here
                                                                                                            for r in range(0, 2):
                                                                                                                if (size_2dL_L9[r] != 0):
                                                                                                                    xyz_L9 = xyz_2dL_L9[r]
                                                                                                                    if (FPS_2dL_L9[r]==1):
                                                                                                                        centerXYZ, centerXYZ_size = selectAfromB(xyz_L9, new_xyz[i,:,:].unsqueeze(0)) # group source
                                                                                                                        if(centerXYZ_size != 0):
                                                                                                                            group_from_xyz = xyz_L8 # group source
                                                                                                                            group_from_xyz_N = group_from_xyz.size(1)
                                                                                                                            idx = torch.cuda.IntTensor(1, centerXYZ_size, nsample, device=xyz.device).zero_()
                                                                                                                            pointnet2_cuda.ball_query_wrapper(1, group_from_xyz_N, centerXYZ_size, radius, nsample, centerXYZ, group_from_xyz, idx)
                                                                                                                            indx_batch = torch.cat((indx_batch, idx), dim=1)
                                                                                                                        else:
                                                                                                                            continue
                                                                                                                    else:#deep 10
                                                                                                                        size_2dL_L10, xyz_2dL_L10, FPS_2dL_L10 = part2_and_count(xyz_L9, FPS_th,9) # actually the FPS_th2 is useless here
                                                                                                                        for s in range(0, 2):
                                                                                                                            if (size_2dL_L10[s] != 0):
                                                                                                                                xyz_L10 = xyz_2dL_L10[s]
                                                                                                                                centerXYZ, centerXYZ_size = selectAfromB(xyz_L10, new_xyz[i,:,:].unsqueeze(0)) # group source
                                                                                                                                if(centerXYZ_size != 0):
                                                                                                                                    group_from_xyz = xyz_L9 # group source
                                                                                                                                    group_from_xyz_N = group_from_xyz.size(1)
                                                                                                                                    idx = torch.cuda.IntTensor(1, centerXYZ_size, nsample, device=xyz.device).zero_()
                                                                                                                                    pointnet2_cuda.ball_query_wrapper(1, group_from_xyz_N, centerXYZ_size, radius, nsample, centerXYZ, group_from_xyz, idx)
                                                                                                                                    indx_batch = torch.cat((indx_batch, idx), dim=1)
                                                                                                                                else:
                                                                                                                                    continue
                                                                                                                            
                            else:
                                continue
                else:
                    continue
            # if(i==23):
            #     pdb.set_trace()
            indx_all = torch.cat((indx_all, indx_batch), dim=0)
            # print('batch=', i, ', indx_batch.size=', indx_batch.size(1))
        # groupXyz_indx = find_indices(xyz, groupXyz, B)
            # pdb.set_trace()
    except RuntimeError:
            print(RuntimeError)
            pdb.set_trace()

    return indx_all

class BallQuery(Function):
    @staticmethod
    def forward(ctx, radius: float, nsample: int, xyz: torch.Tensor, new_xyz: torch.Tensor) -> torch.Tensor:
        """
        :param ctx:
        :param radius: float, radius of the balls
        :param nsample: int, maximum number of features in the balls
        :param xyz: (B, N, 3) xyz coordinates of the features
        :param new_xyz: (B, npoint, 3) centers of the ball query
        :return:
            idx: (B, npoint, nsample) tensor with the indicies of the features that form the query balls
        """
        assert new_xyz.is_contiguous()
        assert xyz.is_contiguous()
        # pdb.set_trace()
        B, N, _ = xyz.size()
        npoint = new_xyz.size(1)

        if(npoint == 256):
            group_th = 64
            idx = Tree_group_depth10_config(xyz, nsample, new_xyz, radius, group_th)
            idx=idx.int()
        else:
            idx = torch.cuda.IntTensor(B, npoint, nsample, device=xyz.device).zero_()
            pointnet2_cuda.ball_query_wrapper(B, N, npoint, radius, nsample, new_xyz, xyz, idx)
        # pdb.set_trace()
        return idx

    @staticmethod
    def backward(ctx, a=None):
        return None, None, None, None


ball_query = BallQuery.apply


class QueryAndGroup(nn.Module):
    def __init__(self, radius: float, nsample: int,
                 relative_xyz=True,
                 normalize_dp=False,
                 normalize_by_std=False,
                 normalize_by_allstd=False,
                 normalize_by_allstd2=False,
                 return_only_idx=False,
                 **kwargs
                 ):
        """[summary]

        Args:
            radius (float): radius of ball
            nsample (int): maximum number of features to gather in the ball
            use_xyz (bool, optional): concate xyz. Defaults to True.
            ret_grouped_xyz (bool, optional): [description]. Defaults to False.
            normalize_dp (bool, optional): [description]. Defaults to False.
        """
        super().__init__()
        self.radius, self.nsample = radius, nsample
        self.normalize_dp = normalize_dp
        self.normalize_by_std = normalize_by_std
        self.normalize_by_allstd = normalize_by_allstd
        self.normalize_by_allstd2 = normalize_by_allstd2
        assert self.normalize_dp + self.normalize_by_std + self.normalize_by_allstd < 2   # only nomalize by one method
        self.relative_xyz = relative_xyz
        self.return_only_idx = return_only_idx

    def forward(self, query_xyz: torch.Tensor, support_xyz: torch.Tensor, features: torch.Tensor = None) -> Tuple[
        torch.Tensor]:
        """
        :param query_xyz: (B, npoint, 3) xyz coordinates of the features
        :param support_xyz: (B, N, 3) centroids
        :param features: (B, C, N) descriptors of the features
        :return:
            new_features: (B, 3 + C, npoint, nsample)
        """

        idx = ball_query(self.radius, self.nsample, support_xyz, query_xyz)

        if self.return_only_idx:
            return idx
        xyz_trans = support_xyz.transpose(1, 2).contiguous()
        grouped_xyz = grouping_operation(xyz_trans, idx)  # (B, 3, npoint, nsample)
        if self.relative_xyz:
            grouped_xyz = grouped_xyz - query_xyz.transpose(1, 2).unsqueeze(-1)  # relative position
            if self.normalize_dp:
                grouped_xyz /= self.radius
        grouped_features = grouping_operation(features, idx) if features is not None else None
        return grouped_xyz, grouped_features


class GroupAll(nn.Module):
    def __init__(self, ):
        super().__init__()

    def forward(self, new_xyz: torch.Tensor, xyz: torch.Tensor, features: torch.Tensor = None):
        """
        :param xyz: (B, N, 3) xyz coordinates of the features
        :param new_xyz: ignored
        :param features: (B, C, N) descriptors of the features
        :return:
            new_features: (B, C + 3, 1, N)
        """
        grouped_xyz = xyz.transpose(1, 2).unsqueeze(2)
        grouped_features = features.unsqueeze(2) if features is not None else None
        return grouped_xyz, grouped_features


class KNNGroup(nn.Module):
    def __init__(self, nsample: int,
                 relative_xyz=True,
                 normalize_dp=False,
                 return_only_idx=False,
                 **kwargs
                 ):
        """[summary]

        Args:
            nsample (int): maximum number of features to gather in the ball
            use_xyz (bool, optional): concate xyz. Defaults to True.
            ret_grouped_xyz (bool, optional): [description]. Defaults to False.
            normalize_dp (bool, optional): [description]. Defaults to False.
        """
        super().__init__()
        self.nsample = nsample
        self.knn = KNN(nsample, transpose_mode=True)
        self.relative_xyz = relative_xyz
        self.normalize_dp = normalize_dp
        self.return_only_idx = return_only_idx

    def forward(self, query_xyz: torch.Tensor, support_xyz: torch.Tensor, features: torch.Tensor = None) -> Tuple[
        torch.Tensor]:
        """
        :param query_xyz: (B, N, 3) xyz coordinates of the features
        :param support_xyz: (B, npoint, 3) centroids
        :param features: (B, C, N) descriptors of the features
        :return:
            new_features: (B, 3 + C, npoint, nsample)
        """
        _, idx = self.knn(support_xyz, query_xyz)
        if self.return_only_idx:
            return idx
        idx = idx.int()
        xyz_trans = support_xyz.transpose(1, 2).contiguous()
        grouped_xyz = grouping_operation(xyz_trans, idx)  # (B, 3, npoint, nsample)
        if self.relative_xyz:
            grouped_xyz -= query_xyz.transpose(1, 2).unsqueeze(-1)  # relative position
        if self.normalize_dp:
            grouped_xyz /= torch.amax(torch.sqrt(torch.sum(grouped_xyz**2, dim=1)), dim=(1, 2)).view(-1, 1, 1, 1)
        if features is not None:
            grouped_features = grouping_operation(features, idx)
            return grouped_xyz, grouped_features
        else:
            return grouped_xyz, None


def get_aggregation_feautres(p, dp, f, fj, feature_type='dp_fj'):
    if feature_type == 'dp_fj':
        fj = torch.cat([dp, fj], 1)
    elif feature_type == 'dp_fj_df':
        df = fj - f.unsqueeze(-1)
        fj = torch.cat([dp, fj, df], 1)
    elif feature_type == 'pi_dp_fj_df':
        df = fj - f.unsqueeze(-1)
        fj = torch.cat([p.transpose(1, 2).unsqueeze(-1).expand(-1, -1, -1, df.shape[-1]), dp, fj, df], 1)
    elif feature_type == 'dp_df':
        df = fj - f.unsqueeze(-1)
        fj = torch.cat([dp, df], 1)
    return fj


def create_grouper(group_args):
    group_args_copy = copy.deepcopy(group_args)
    method = group_args_copy.pop('NAME', 'ballquery')
    radius = group_args_copy.pop('radius', 0.1)
    nsample = group_args_copy.pop('nsample', 20)

    logging.info(group_args)
    if nsample is not None:
        if method == 'ballquery':
            grouper = QueryAndGroup(radius, nsample, **group_args_copy)
        elif method == 'knn':
            grouper = KNNGroup(nsample,  **group_args_copy)
    else:
        grouper = GroupAll()
    return grouper


if __name__ == "__main__":
    import time

    B, C, N = 2, 3, 40960
    K = 16
    device = 'cuda'
    points = torch.randn([B, N, C], device=device, dtype=torch.float)
    print(points.shape, '\n', points)

    # --------------- debug downsampling
    from openpoints.models.layers.layer3d import RandomSample, random_sample, furthest_point_sample

    npoints = 10000
    # rs = RandomSample(num_to_sample=npoints)
    # query, _= rs(points)
    idx = random_sample(points, npoints)
    # torch gather is faster then operation gather. 
    query = torch.gather(points, 1, idx.unsqueeze(-1).expand(-1, -1, 3))
    print(query.shape, '\n', query)

    idx = furthest_point_sample(points, npoints).to(torch.int64)
    query = torch.gather(points, 1, idx.unsqueeze(-1).expand(-1, -1, 3))
    print(query.shape, '\n', query)

    # # --------------- debug KNN
    # knn = KNN(k=K, transpose_mode=True)
    # # knn to get the neighborhood

    # # compare time usage.
    # st = time.time()
    # for _ in range(100):
    #     _, knnidx = knn(points, query) # B G M
    #     idx_base = torch.arange(0, B, device=points.device).view(-1, 1, 1) * N
    #     idx = knnidx + idx_base
    #     idx = idx.view(-1)
    #     neighborhood = points.view(B * N, -1)[idx, :]
    #     neighborhood = neighborhood.view(B, npoints, K, 3).contiguous()
    #     # normalize
    #     neighborhood1 = neighborhood - query.unsqueeze(2)
    # print(time.time() - st)
    # # print(neighborhood1.shape, '\n', neighborhood1)

    # knngroup = KNNGroup(K)
    # # KNN Group is faster then above torch indexing when warpped in class.  
    # st = time.time()
    # for _ in range(100):
    #     neighborhood2 = knngroup(query, points)
    # print(time.time() - st)
    # # print(neighborhood2.shape, '\n', neighborhood2)
    # flag = torch.allclose(neighborhood1, neighborhood2.permute(0, 2, 3, 1))
    # print(flag)

    # ------------- debug ball query
    query_group = QueryAndGroup(0.1, K)

    st = time.time()
    for _ in range(100):
        # ball querying is 40 times faster then KNN 
        features = query_group(query, points)
    print(time.time() - st)
    print(features.shape)
