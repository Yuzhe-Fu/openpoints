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
                raise ValueError("Can only specify ratio or num_to_sample or subsampling_param, not several !")
            self._num_to_sample = num_to_sample

        elif ratio is not None:
            self._ratio = ratio

        elif subsampling_param is not None:
            self._subsampling_param = subsampling_param

        else:
            raise Exception('At least ["ratio, num_to_sample, subsampling_param"] should be defined')

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
        idx = torch.randint(0, N, (B, self._get_num_to_sample(N)), device=xyz.device)
        sampled_xyz = torch.gather(xyz, 1, idx.unsqueeze(-1).expand(-1, -1, 3))
        # sampled_feature = torch.gather(feature, 2, idx.unsqueeze(1).repeat(1, C, 1))
        return sampled_xyz, idx


def random_sample(xyz, npoint):
    B, N, _ = xyz.shape
    idx = torch.randint(0, N, (B, npoint), device=xyz.device)
    return idx


def quanti8(x):
   return torch.trunc(x*(2**8))/2**8

def quanti32(x):
   return torch.trunc(x*(2**32))/2**32


def del_tensor_ele_n(arr, index, n):
    # input is 3-dimention
    arr1 = arr[:,0:index,:]
    arr2 = arr[:,index+n:,:]
    return torch.cat((arr1,arr2),dim=1)

# def spar_tensor(arr, scale):
#     B, N, _ = arr.size()
#     itear = int(N/scale)
#     temp = arr
#     for i in range(itear-1):  # i from 0 to 510
#         temp = del_tensor_ele_n(temp, i+1,1)
#     temp = temp[:,0:itear,:]
#     return temp

def spar_tensor(arr, scale, starPoint=0):
    return arr[:,starPoint::scale,:]

def my_fps(xyz, npoint):
    B, N, _ = xyz.size()
    output = torch.cuda.IntTensor(B, npoint)
    temp = torch.cuda.FloatTensor(B, N).fill_(1e10)
    pointnet2_cuda.furthest_point_sampling_wrapper(B, N, npoint, xyz, temp, output)
    return output

def domain_fps_16block(xyz, scale, npoint):
    # first, sparse the xyz input to 1/S
    xyz_0_temp = spar_tensor(xyz,scale) #(64,512,3)

    # pass the sparsed xyz into fps get the sparsed output
    spar_out = my_fps(xyz_0_temp, int(npoint/(scale)))#(64,256)

    # classfy the output into 8 domain and count it number
    B, N = spar_out.size()
    spar_out = spar_out.to(torch.int64)

    query = torch.gather(xyz_0_temp, 1, spar_out.unsqueeze(-1).expand(-1, -1, 3)) #(64,256,3)
    query_bit = query < 0
    query_xless05 = torch.abs(query[:,:,0])<0.5
    query0 = 1*query_bit[:,:,0]+2*query_bit[:,:,1]+4*query_bit[:,:,2]+8*query_xless05
    domain = []

    for b in range(B):
        domain_temp = []
        for i in range(16):
            domain_temp.append(torch.nonzero(query0[b,:]==i).size()[0]*scale) # multiple the number with scale
        domain.append(domain_temp)

    # according the updated number, devide the xyz into 8 domain, and fps respectively
    xyz_0_temp = xyz < 0
    xyz_0_xless05 = torch.abs(xyz[:,:,0])<0.5

    xyz_0_temp = 1*xyz_0_temp[:,:,0]+2*xyz_0_temp[:,:,1]+4*xyz_0_temp[:,:,2]+8*xyz_0_xless05

    xyz_splited_out = torch.tensor([],device='cuda')
    for b in range(B):
        out = torch.tensor([],device='cuda')
        for i in range(16):
            # out_temp_temp = torch.tensor([],device='cuda')
            if(domain[b][i] != 0):
                out_temp_temp = []

                indx = torch.nonzero(xyz_0_temp[b,:]==i).reshape(1,-1)
                indx = indx.to(torch.int64)
                tensor_temp = (torch.gather(torch.reshape(xyz[b,:], [1,-1,3]), 1, indx.unsqueeze(-1).expand(-1, -1, 3)))

                out_temp = my_fps(tensor_temp , int(domain[b][i])).reshape(-1)
                # pdb.set_trace()
                for o in out_temp:
                    out_temp_temp.append(indx[0, o].item())
                    # out_temp_temp = torch.cat((out_temp_temp, indx[0, i].reshape(1)), dim=0)
                out = torch.cat((out, torch.tensor(out_temp_temp, device='cuda')), dim=0)
        xyz_splited_out = torch.cat((xyz_splited_out, out.reshape([1,-1])), dim=0)
    return xyz_splited_out.to(torch.int32)

def domain_fps_8block_sparto16(xyz, scale, npoint):
    # first, sparse the xyz input to 1/S
    xyz_0_temp = spar_tensor(xyz,scale, 0) #(64,512,3)

    # pass the sparsed xyz into fps get the sparsed output
    spar_out = my_fps(xyz_0_temp[:,0::2,:], int(npoint/(2*scale)))*2 #(64,256) *2有点问题 #FIXME
    spar_out = torch.cat((spar_out, (my_fps(xyz_0_temp[:,1::2,:], int(npoint/(2*scale)))*2+1)), dim=1)
    # pdb.set_trace()
    # classfy the output into 8 domain and count it number
    B, N = spar_out.size()
    spar_out = spar_out.to(torch.int64)

    query = torch.gather(xyz_0_temp, 1, spar_out.unsqueeze(-1).expand(-1, -1, 3)) #(64,256,3)
    query = query < 0
    query0 = 1*query[:,:,0]+2*query[:,:,1]+4*query[:,:,2]
    domain = []
    for b in range(B):
        domain_temp = []
        for i in range(8):
            domain_temp.append(torch.nonzero(query0[b,:]==i).size()[0]*scale) # multiple the number with scale
        domain.append(domain_temp)

    # according the updated number, devide the xyz into 8 domain, and fps respectively
    xyz_0_temp = xyz < 0
    xyz_0_temp = 1*xyz_0_temp[:,:,0]+2*xyz_0_temp[:,:,1]+4*xyz_0_temp[:,:,2]

    xyz_splited_out = torch.tensor([],device='cuda')
    for b in range(B):
        out = torch.tensor([],device='cuda')
        for i in range(8):
            if(domain[b][i] != 0):
                out_temp_temp = []
                indx = torch.nonzero(xyz_0_temp[b,:]==i).reshape(1,-1)
                indx = indx.to(torch.int64)
                tensor_temp = (torch.gather(torch.reshape(xyz[b,:], [1,-1,3]), 1, indx.unsqueeze(-1).expand(-1, -1, 3)))
                if indx.size()[1]>64:
                    out_temp = my_fps(tensor_temp[:,0::2,:] , int(domain[b][i]/2))*2
                    out_temp = torch.cat((out_temp, (my_fps(tensor_temp[:,1::2,:] , int(domain[b][i]/2))*2+1)), dim=1)
                    out_temp = out_temp.reshape(-1)
                else:
                    out_temp = my_fps(tensor_temp , int(domain[b][i])).reshape(-1)
                # pdb.set_trace()
                try:
                    for o in out_temp:
                        out_temp_temp.append(indx[0, int(o)].item())
                except Exception as e:
                    pdb.set_trace()
                    # out_temp_temp = torch.cat((out_temp_temp, indx[0, i].reshape(1)), dim=0)
                out = torch.cat((out, torch.tensor(out_temp_temp, device='cuda')), dim=0)
        
        xyz_splited_out = torch.cat((xyz_splited_out, out.reshape([1,-1])), dim=0)

    return xyz_splited_out.to(torch.int32)



def domain_fps_8block(xyz, scale, npoint):
    # first, sparse the xyz input to 1/S
    xyz_0_temp = spar_tensor(xyz,scale) #(64,512,3)

    # pass the sparsed xyz into fps get the sparsed output
    spar_out = my_fps(xyz_0_temp, int(npoint/(scale)))#(64,256)

    # classfy the output into 8 domain and count it number
    B, N = spar_out.size()
    spar_out = spar_out.to(torch.int64)

    query = torch.gather(xyz_0_temp, 1, spar_out.unsqueeze(-1).expand(-1, -1, 3)) #(64,256,3)
    query = query < 0
    query0 = 1*query[:,:,0]+2*query[:,:,1]+4*query[:,:,2]
    domain = []
    for b in range(B):
        domain_temp = []
        for i in range(8):
            domain_temp.append(torch.nonzero(query0[b,:]==i).size()[0]*scale) # multiple the number with scale
        domain.append(domain_temp)

    # according the updated number, devide the xyz into 8 domain, and fps respectively
    xyz_0_temp = xyz < 0
    xyz_0_temp = 1*xyz_0_temp[:,:,0]+2*xyz_0_temp[:,:,1]+4*xyz_0_temp[:,:,2]

    xyz_splited_out = torch.tensor([],device='cuda')
    for b in range(B):
        out = torch.tensor([],device='cuda')
        for i in range(8):
            if(domain[b][i] != 0):
                out_temp_temp = []

                indx = torch.nonzero(xyz_0_temp[b,:]==i).reshape(1,-1)
                indx = indx.to(torch.int64)
                tensor_temp = (torch.gather(torch.reshape(xyz[b,:], [1,-1,3]), 1, indx.unsqueeze(-1).expand(-1, -1, 3)))

                out_temp = my_fps(tensor_temp , int(domain[b][i])).reshape(-1)
                # pdb.set_trace()
                for i in out_temp:
                    out_temp_temp.append(indx[0, i].item())
                    # out_temp_temp = torch.cat((out_temp_temp, indx[0, i].reshape(1)), dim=0)
                out = torch.cat((out, torch.tensor(out_temp_temp, device='cuda')), dim=0)
        xyz_splited_out = torch.cat((xyz_splited_out, out.reshape([1,-1])), dim=0)

    return xyz_splited_out.to(torch.int32)

def domain_fps_4block(xyz, scale, npoint):
    # first, sparse the xyz input to 1/S
    xyz_0_temp = spar_tensor(xyz,scale) #(64,512,3)

    # pass the sparsed xyz into fps get the sparsed output
    spar_out = my_fps(xyz_0_temp, int(npoint/(scale)))#(64,256)

    # classfy the output into 4 domain and count it number
    B, N = spar_out.size()
    spar_out = spar_out.to(torch.int64)

    query = torch.gather(xyz_0_temp, 1, spar_out.unsqueeze(-1).expand(-1, -1, 3)) #(64,256,3)
    query = query[:,:,0:2] < 0
    query0 = 1*query[:,:,0]+2*query[:,:,1]
    domain = []
    for b in range(B):
        domain_temp = []
        for i in range(4):
            domain_temp.append(torch.nonzero(query0[b,:]==i).size()[0]*scale) # multiple the number with scale
        domain.append(domain_temp)

    # according the updated number, devide the xyz into 4 domain, and fps respectively
    xyz_0_temp = xyz < 0
    xyz_0_temp = 1*xyz_0_temp[:,:,0]+2*xyz_0_temp[:,:,1]

    xyz_splited_out = torch.tensor([],device='cuda')
    for b in range(B):
        out = torch.tensor([],device='cuda')
        for i in range(4):
            # out_temp_temp = torch.tensor([],device='cuda')
            if(domain[b][i] != 0):
                out_temp_temp = []

                indx = torch.nonzero(xyz_0_temp[b,:]==i).reshape(1,-1)
                indx = indx.to(torch.int64)
                tensor_temp = (torch.gather(torch.reshape(xyz[b,:], [1,-1,3]), 1, indx.unsqueeze(-1).expand(-1, -1, 3)))

                out_temp = my_fps(tensor_temp , int(domain[b][i])).reshape(-1)
                # pdb.set_trace()
                for i in out_temp:
                    out_temp_temp.append(indx[0, i].item())
                    # out_temp_temp = torch.cat((out_temp_temp, indx[0, i].reshape(1)), dim=0)
                out = torch.cat((out, torch.tensor(out_temp_temp, device='cuda')), dim=0)
        xyz_splited_out = torch.cat((xyz_splited_out, out.reshape([1,-1])), dim=0)

    return xyz_splited_out.to(torch.int32)


def domain_fps_2block(xyz, scale, npoint):
    # first, sparse the xyz input to 1/S
    xyz_0_temp = spar_tensor(xyz,scale) #(64,512,3)

    # pass the sparsed xyz into fps get the sparsed output
    spar_out = my_fps(xyz_0_temp, int(npoint/(scale)))#(64,256)

    # classfy the output into 4 domain and count it number
    B, N = spar_out.size()
    spar_out = spar_out.to(torch.int64)

    query = torch.gather(xyz_0_temp, 1, spar_out.unsqueeze(-1).expand(-1, -1, 3)) #(64,256,3)
    query = query[:,:,0] < 0
    query0 = 1*query
    domain = []
    for b in range(B):
        domain_temp = []
        for i in range(2):
            domain_temp.append(torch.nonzero(query0[b,:]==i).size()[0]*scale) # multiple the number with scale
        domain.append(domain_temp)

    # according the updated number, devide the xyz into 4 domain, and fps respectively
    xyz_0_temp = xyz[:,:,0] < 0
    xyz_0_temp = 1*xyz_0_temp[:,:]

    xyz_splited_out = torch.tensor([],device='cuda')
    for b in range(B):
        out = torch.tensor([],device='cuda')
        for i in range(2):
            # out_temp_temp = torch.tensor([],device='cuda')
            if(domain[b][i] != 0):
                out_temp_temp = []

                indx = torch.nonzero(xyz_0_temp[b,:]==i).reshape(1,-1)
                indx = indx.to(torch.int64)
                tensor_temp = (torch.gather(torch.reshape(xyz[b,:], [1,-1,3]), 1, indx.unsqueeze(-1).expand(-1, -1, 3)))

                out_temp = my_fps(tensor_temp , int(domain[b][i])).reshape(-1)
                # pdb.set_trace()
                for i in out_temp:
                    out_temp_temp.append(indx[0, i].item())
                    # out_temp_temp = torch.cat((out_temp_temp, indx[0, i].reshape(1)), dim=0)
                out = torch.cat((out, torch.tensor(out_temp_temp, device='cuda')), dim=0)
        xyz_splited_out = torch.cat((xyz_splited_out, out.reshape([1,-1])), dim=0)

    return xyz_splited_out.to(torch.int32)



def domain_fps_v1 (xyz, scale, npoint):
    # first, sparse the xyz input to 1/S
    xyz_0_temp = spar_tensor(xyz,scale) #(64,512,3)

    # pass the sparsed xyz into fps get the sparsed output
    spar_out = my_fps(xyz_0_temp, int(npoint/(scale)))#(64,256)

    # classfy the output into 8 domain and count it number
    B, N = spar_out.size()
    spar_out = spar_out.to(torch.int64)

    query = torch.gather(xyz_0_temp, 1, spar_out.unsqueeze(-1).expand(-1, -1, 3)) #(64,256,3)
    query = query < 0
    query0 = 1*query[:,:,0]+2*query[:,:,1]+4*query[:,:,2] #(64, 256)

    domain = torch.tensor([],device='cuda')
    # domain = []
    for i in range(8):
        # domain.append((query0 == i).sum(dim=1)).reshape(1,B)
        domain = torch.cat((domain, ((query0 == i).sum(dim=1)).reshape(B,1)), dim=1)
    domain = domain*scale #(64,8)

    # the above is right!
    # according the updated number, devide the xyz into 8 domain, and fps respectively
    xyz_0_temp = xyz < 0
    xyz_0_temp = 1*xyz_0_temp[:,:,0]+2*xyz_0_temp[:,:,1]+4*xyz_0_temp[:,:,2]
    
    # get the gather indx
    out_temp_list = []
    for i in range(8):
        indx = torch.nonzero(xyz_0_temp==i)
        list_in_list = [[] for _ in range(B)]
        for temp in indx:
            list_in_list[temp[0]].append(temp[1])

        for b in range(B):
            appended_list = []
            if len(list_in_list[b]) < npoint:
                if len(list_in_list[b]) == 0:
                    appended_list = [0] * npoint
                else:
                    appended_list = [list_in_list[b][0]] * (npoint-len(list_in_list[b]))
            list_in_list[b].extend(appended_list)
            # pdb.set_trace()
        
        indx = torch.tensor(list_in_list, device='cuda') #(64, 512) for one domain
        tensor_temp = (torch.gather(xyz, 1, indx.unsqueeze(-1).expand(-1, -1, 3)))

        out_temp = my_fps(tensor_temp , int(npoint/2))
        out_temp = torch.gather(indx, 1, out_temp.to(torch.int64))
        out_temp_list.append(out_temp) #[8,64,256]
    
    final_out_tensor = torch.tensor([],device='cuda')
    domain = domain.to(torch.int32)
    for b in range(B):
        final_out_temp = torch.tensor([],device='cuda')
        for i in range(8):
            length = domain[b,i]
            # pdb.set_trace()
            if domain[b,i] != 0:
                final_out_temp = torch.cat((final_out_temp, out_temp_list[i][b][0:domain[b,i]]),dim=0)
        final_out_tensor = torch.cat((final_out_tensor, final_out_temp.reshape([1,-1])),dim=0).to(torch.int32)

    return final_out_tensor


def fps_myown_v1 (xyz, bnum, npoint):
    B,N,_ = xyz.size()
    # block_out = torch.tensor([],device='cuda')
    block_out = torch.cuda.IntTensor(B, npoint)

    for i in range(bnum):
        # block_out = torch.cat((block_out, (my_fps(xyz[:,i::bnum,:], int(npoint/bnum))*bnum+i)), dim=1)
        block_out[:,i::bnum] = (my_fps(xyz[:,i::bnum,:], int(npoint/bnum))*bnum+i).to(torch.int32)

    # block_out = block_out.to(torch.int32)
    return block_out

def fps_myown_seg (xyz, bnum, npoint):
    B,N,_ = xyz.size()
    # block_out = torch.tensor([],device='cuda')
    block_out = torch.cuda.IntTensor(B, npoint)
    base = (N/bnum)
    base1 = (npoint/bnum)

    for i in range(bnum):
        block_out[:,int(i*base1):int((i+1)*base1)] = (my_fps(xyz[:,int(i*base):int((i+1)*base),:], int(npoint/bnum)) + int(i*base)).to(torch.int32)
    
    # block_out = block_out.to(torch.int32)
    return block_out


def domain_fps_27block(xyz, scale, npoint):
    # first, sparse the xyz input to 1/S
    xyz_0_temp = spar_tensor(xyz,scale) #(64,512,3)

    # pass the sparsed xyz into fps get the sparsed output
    spar_out = my_fps(xyz_0_temp, int(npoint/(scale)))#(64,256)

    # classfy the output into 27 domain and count it number
    B, N = spar_out.size()
    spar_out = spar_out.to(torch.int64)

    query = torch.gather(xyz_0_temp, 1, spar_out.unsqueeze(-1).expand(-1, -1, 3)) #(64,256,3)
    query_biggerP33 = query > 0.3
    query_lessNeg33 = query < -0.3

    query_all = 1*query_biggerP33[:,:,0]+2*query_biggerP33[:,:,1]+4*query_biggerP33[:,:,2]+ \
        8*query_lessNeg33[:,:,0]+16*query_lessNeg33[:,:,1]+32*query_lessNeg33[:,:,2]
    
    domain = []
    for b in range(B):
        domain_temp = []
        for i in range(64):
            domain_temp.append(torch.nonzero(query_all[b,:]==i).size()[0]*scale) # multiple the number with scale
        domain.append(domain_temp)

    # according the updated number, devide the xyz into 8 domain, and fps respectively

    xyz_biggerP33 = xyz > 0.3
    xyz_lessNeg33 = xyz < -0.3

    xyz_0_temp = 1*xyz_biggerP33[:,:,0]+2*xyz_biggerP33[:,:,1]+4*xyz_biggerP33[:,:,2]+ \
        8*xyz_lessNeg33[:,:,0]+16*xyz_lessNeg33[:,:,1]+32*xyz_lessNeg33[:,:,2]
    
    xyz_splited_out = torch.tensor([],device='cuda')
    for b in range(B):
        out = torch.tensor([],device='cuda')
        for i in range(64):
            if(domain[b][i] != 0):
                out_temp_temp = []

                indx = torch.nonzero(xyz_0_temp[b,:]==i).reshape(1,-1)
                indx = indx.to(torch.int64)
                tensor_temp = (torch.gather(torch.reshape(xyz[b,:], [1,-1,3]), 1, indx.unsqueeze(-1).expand(-1, -1, 3)))

                out_temp = my_fps(tensor_temp , int(domain[b][i])).reshape(-1)
                # pdb.set_trace()
                for i in out_temp:
                    out_temp_temp.append(indx[0, i].item())
                    # out_temp_temp = torch.cat((out_temp_temp, indx[0, i].reshape(1)), dim=0)
                out = torch.cat((out, torch.tensor(out_temp_temp, device='cuda')), dim=0)
        xyz_splited_out = torch.cat((xyz_splited_out, out.reshape([1,-1])), dim=0)

    return xyz_splited_out.to(torch.int32)


# ====================================================================================================================================
# ====================================================================================================================================
# ################ Multi-Stream Block-Wise Prediction// which is tcas-ii version
# blockNum = 2,4,6,16,32
# PMS = 1,2,...,8,... (< scale)
# ====================================================================================================================================
# ====================================================================================================================================
def OneStream_block_wise_Predict(xyz, npoint, blockNum, scale, bias):
        # first, sparse the xyz input to 1/S
    xyz_0_temp = spar_tensor(xyz, scale, bias) #(64,512,3)

    # pass the sparsed xyz into fps get the sparsed output
    spar_out = my_fps(xyz_0_temp, int(npoint/scale)) 
    # classfy the output into 8 domain and count it number
    B, _ = spar_out.size()
    spar_out = spar_out.to(torch.int64)

    query = torch.gather(xyz_0_temp, 1, spar_out.unsqueeze(-1).expand(-1, -1, 3)) #(64,256,3)

    xyz_0_temp = query < 0
    xyz_0_xless05 = torch.abs(query)<0.5
    if blockNum == 2:
        xyz_0_temp = 1*xyz_0_temp[:,:,0]
    elif blockNum == 4:
        xyz_0_temp = 1*xyz_0_temp[:,:,0]+2*xyz_0_temp[:,:,1]
    elif blockNum == 8:
        xyz_0_temp = 1*xyz_0_temp[:,:,0]+2*xyz_0_temp[:,:,1]+4*xyz_0_temp[:,:,2]
    elif blockNum == 16:
        xyz_0_temp = 1*xyz_0_temp[:,:,0]+2*xyz_0_temp[:,:,1]+4*xyz_0_temp[:,:,2]+8*xyz_0_xless05[:,:,0]
    else: # now Num is 32
        xyz_0_temp = 1*xyz_0_temp[:,:,0]+2*xyz_0_temp[:,:,1]+4*xyz_0_temp[:,:,2]+8*xyz_0_xless05[:,:,0]+16*xyz_0_xless05[:,:,1]

    finalPredictNuminBlock = []
    for b in range(B):
        domain_temp = []
        for i in range(blockNum):
            domain_temp.append(torch.nonzero(xyz_0_temp[b,:]==i).size()[0]) # multiple the number with scale
        finalPredictNuminBlock.append(domain_temp)

    return finalPredictNuminBlock

def MultiStream_block_wise_Predict(xyz, npoint, blockNum, scale, PMS):
    # first, sparse the xyz input to 1/S
    B, N, _ = xyz.size()
    finalPredictNuminBlock = [[0 for _ in range(blockNum)] for _ in range(B)] 
    for i in range(PMS):
        temp = OneStream_block_wise_Predict(xyz, npoint, blockNum, scale, i)
        finalPredictNuminBlock = [[a+b for a, b in zip(sublist1, sublist2)] for sublist1, sublist2 in zip(finalPredictNuminBlock, temp)]
    for b in range(B):
        for i in range(blockNum):
            finalPredictNuminBlock[b][i] = round(finalPredictNuminBlock[b][i] * scale/(PMS))
        sumOfPN = sum(finalPredictNuminBlock[b])
        if sumOfPN != npoint:
            dif = npoint - sumOfPN
            max_value = max(finalPredictNuminBlock[b])
            max_position = finalPredictNuminBlock[b].index(max_value)
            finalPredictNuminBlock[b][max_position] = finalPredictNuminBlock[b][max_position] + dif
    return finalPredictNuminBlock


# ====================================================================================================================================
# ====================================================================================================================================
# ################ Multi-Stream Block-Wise FPS
# blockNum = 2,4,6,16,32
# BMS = 1,2,...,8,...
# ====================================================================================================================================
# ====================================================================================================================================
def MultiStream_block_wise_FPS(xyz, predict_list, blockNum, BMS):
    xyz_0_temp = xyz < 0
    xyz_0_xless05 = torch.abs(xyz)<0.5
    if blockNum == 2:
        xyz_0_temp = 1*xyz_0_temp[:,:,0]
    elif blockNum == 4:
        xyz_0_temp = 1*xyz_0_temp[:,:,0]+2*xyz_0_temp[:,:,1]
    elif blockNum == 8:
        xyz_0_temp = 1*xyz_0_temp[:,:,0]+2*xyz_0_temp[:,:,1]+4*xyz_0_temp[:,:,2]
    elif blockNum == 16:
        xyz_0_temp = 1*xyz_0_temp[:,:,0]+2*xyz_0_temp[:,:,1]+4*xyz_0_temp[:,:,2]+8*xyz_0_xless05[:,:,0]
    else: # now Num is 32
        xyz_0_temp = 1*xyz_0_temp[:,:,0]+2*xyz_0_temp[:,:,1]+4*xyz_0_temp[:,:,2]+8*xyz_0_xless05[:,:,0]+16*xyz_0_xless05[:,:,1]

    xyz_splited_out = torch.tensor([],device='cuda')
    B, N, _ = xyz.size()
    Th = round(N/blockNum/BMS)
    for b in range(B):
        out = torch.tensor([],device='cuda')
        # pdb.set_trace()
        for i in range(blockNum):
            if(predict_list[b][i] != 0):
                out_temp_temp = []
                indx = torch.nonzero(xyz_0_temp[b,:]==i).reshape(1,-1)
                indx = indx.to(torch.int64)
                tensor_temp = (torch.gather(torch.reshape(xyz[b,:], [1,-1,3]), 1, indx.unsqueeze(-1).expand(-1, -1, 3)))
                if BMS == 1:
                    out_temp = my_fps(tensor_temp , int(predict_list[b][i])).reshape(-1)
                else:
                    if indx.size()[1] <= Th or indx.size()[1] <= BMS or predict_list[b][i] < (Th/2): # 
                        out_temp = my_fps(tensor_temp , int(predict_list[b][i])).reshape(-1)
                    elif BMS == 2:
                        half_num = int(predict_list[b][i]/2)
                        out_temp = my_fps(tensor_temp[:,0::2,:] , half_num)*2
                        out_temp = torch.cat((out_temp, my_fps(tensor_temp[:,1::2,:] , int(predict_list[b][i]-half_num))*2+1), dim=1)
                        out_temp = out_temp.reshape(-1)
                    else: # BMS >= 3
                        th = int(predict_list[b][i]/BMS)
                        out_temp = my_fps(tensor_temp[:,0::BMS,:], th)*BMS
                        for bms in range(1, BMS-1):
                            out_temp = torch.cat((out_temp, my_fps(tensor_temp[:,bms::BMS,:], th)*BMS+bms), dim=1)
                        out_temp = torch.cat((out_temp, my_fps(tensor_temp[:,(BMS-1)::BMS,:] , int(predict_list[b][i]-(BMS-1)*th))*BMS+(BMS-1)), dim=1)
                        out_temp = out_temp.reshape(-1)
                        
                # pdb.set_trace()
                # try:
                for o in out_temp:
                    out_temp_temp.append(indx[0, int(o)].item())
                # except Exception as e:
                    # pdb.set_trace()
                    # continue
                    # out_temp_temp = torch.cat((out_temp_temp, indx[0, i].reshape(1)), dim=0)
                out = torch.cat((out, torch.tensor(out_temp_temp, device='cuda')), dim=0)
        xyz_splited_out = torch.cat((xyz_splited_out, out.reshape([1,-1])), dim=0)
    return xyz_splited_out.to(torch.int32)

# ====================================================================================================================================
# ====================================================================================================================================
# ################ adaptive Block-Wise FPS
# ====================================================================================================================================
# ====================================================================================================================================
def MultiSteram_blockwise_fps_test(xyz, npoint, blockNum, scale, PMS=1, BMS=1):
    finalPredictNuminBlock = MultiStream_block_wise_Predict(xyz, npoint, blockNum, scale, PMS)
    xyz_splited_out = MultiStream_block_wise_FPS(xyz, finalPredictNuminBlock, blockNum, BMS)
    return xyz_splited_out

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
        blockNum = 16
        scale = 16
        PMS=1
        BMS=1

        if npoint==512:
            output = MultiSteram_blockwise_fps_test(xyz, npoint, blockNum, scale, PMS, BMS)# actually spar 16 in algrithom 
            # output = domain_fps_8block_sparto16(xyz, 8, npoint) # actually spar 16 in algrithom
            # output = fps_myown_seg(xyz, 16, npoint)
        elif npoint == 256:
            output = domain_fps_8block(xyz, 8, npoint)
            # output = fps_myown_seg(xyz, 8, npoint)
        elif npoint == 128:
            output = domain_fps_4block(xyz, 4, npoint)
            # output = fps_myown_seg(xyz, 4, npoint)
        elif npoint == 64:
            output = domain_fps_2block(xyz, 2, npoint)
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

        pointnet2_cuda.gather_points_wrapper(B, C, N, npoint, features, idx, output)

        ctx.for_backwards = (idx, C, N)
        return output

    @staticmethod
    def backward(ctx, grad_out):    # todo: understand this part. why needs this backward??
        idx, C, N = ctx.for_backwards
        B, npoint = idx.size()

        grad_features = torch.zeros([B, C, N], dtype=torch.float, device=grad_out.device, requires_grad=True)
        grad_out_data = grad_out.data.contiguous()
        pointnet2_cuda.gather_points_grad_wrapper(B, C, N, npoint, grad_out_data, idx, grad_features.data)
        return grad_features, None


gather_operation = GatherOperation.apply
# mark: torch gather is even faster. sampled_xyz = torch.gather(points, 1, idx.unsqueeze(-1).expand(-1, -1, 3))


def fps(data, number):
    '''
        data B N C
        number int
    '''
    fps_idx = furthest_point_sample(data[:,:, :3].contiguous(), number) 
    # fps_data = gather_operation(data.transpose(1, 2).contiguous(), fps_idx).transpose(1,2).contiguous()
    fps_data = torch.gather(data, 1, fps_idx.unsqueeze(-1).long().expand(-1, -1, data.shape[-1]))
    return fps_data


if __name__ == '__main__':
    import time 
    
    B, C, N = 2, 3, 10000
    K=16
    device = 'cuda'
    points = torch.randn([B, N, 3], device=device, dtype=torch.float)
    print(points.shape, '\n', points)
    
    nsample = 4096
    idx = furthest_point_sample(points, nsample)
    
    st = time.time()
    for _ in range(100): 
        query1 = torch.gather(points, 1, idx.long().unsqueeze(-1).expand(-1, -1, 3))
    print(time.time() - st)
    print(query1.shape)

    st = time.time()
    for _ in range(100):
        query2 = gather_operation(points.transpose(1, 2).contiguous(), idx).transpose(1,2).contiguous()
    print(time.time() - st)
    print(query2.shape)

    print(torch.allclose(query1, query2))
