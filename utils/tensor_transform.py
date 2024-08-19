import torch
import torch.nn as nn
def feature_seperator(input,enc_in):
    """
    输入:(32,12,3584)
    输出:(32,7,12,512)
    """
    B,N,Dim = input.size()
    output = input.view(B,N,enc_in,-1).transpose(1,2)
    return output

def transpose_N_Dim(input):
    output = input.transpose(2,3)
    return output

def flatten_patch(input):
    flatten = nn.Flatten(start_dim=-2)
    output = flatten(input)
    return output

if __name__ == '__main__':
    """
    验证feature_seperator正确性
    输入:(2,2,4)
    输出:(2,2,2,2)
    输入:[[[1,2,3,4],[5,6,7,8]],
         [9,10,11,12],[13,14,15,16]]
    期望输出:[[[[1,2],[5,6]],[[3,4],[7,8]]],
            [[[9,10],[13,14]],[[11,12],[15,16]]]]
    """
    input = torch.tensor([[[1, 2, 3, 4], [5, 6, 7, 8]], [[9, 10, 11, 12], [13, 14, 15, 16]]], dtype=torch.float32)
    output = feature_seperator(input, 2)
    print(output)

    """
    验证transpose_N_Dim正确性
    输入:(2,2,2,2)
    输出:(2,2,2,2)
    输入:[[[[1,2],[5,6]],[[3,4],[7,8]]],
            [[[9,10],[13,14]],[[11,12],[15,16]]]]
    期望输出:[[[[1,5],[2,6]],[[3,7],[4,8]]],
            [[[9,13],[10,14]],[[11,15],[12,16]]]]
    """
    input = torch.tensor([[[[1,2],[5,6]],[[3,4],[7,8]]],
            [[[9,10],[13,14]],[[11,12],[15,16]]]], dtype=torch.float32)
    output = transpose_N_Dim(input)
    print(output)

    """
    验证flatten_patch正确性
    输入:(2,2,2,2)
    输出:(2,2,4)
    输入:[[[[1,5],[2,6]],[[3,7],[4,8]]],
            [[[9,13],[10,14]],[[11,15],[12,16]]]]
    期望输出:[[[1,2,5,6],[3,4,7,8]],[[9,10,13,14],[11,12,15,16]]]
    """
    input = torch.tensor([[[[1,5],[2,6]],[[3,7],[4,8]]],
            [[[9,13],[10,14]],[[11,15],[12,16]]]], dtype=torch.float32)
    output = flatten_patch(input)
    print(output)