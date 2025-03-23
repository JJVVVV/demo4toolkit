import numpy as np
import torch
import torch.nn.functional as F


def generate_distribution(input_ids, vocab_size, min_main_score=0.7):
    one_hot = F.one_hot(input_ids, num_classes=vocab_size)
    main_score = np.random.rand(1).item() * (1 - min_main_score) + min_main_score
    # 生成符合正态分布的随机数
    dis = torch.abs(torch.randn((input_ids.shape[0], input_ids.shape[1], vocab_size), device=input_ids.device))
    # 对随机数进行归一化，使它们的和为x
    dis *= abs(one_hot - 1)
    dis = F.normalize(dis, dim=-1, p=1) * (1 - main_score)
    dis += one_hot * main_score
    return dis


def generate_distribution2(input_ids, vocab_size, min_main_score=0.7):
    one_hot = F.one_hot(input_ids, vocab_size)
    main_score = (torch.rand(*one_hot.shape[:-1], dtype=torch.float32, device=input_ids.device) * (1 - min_main_score) + min_main_score).unsqueeze(-1)
    # 生成符合正态分布的随机数
    dis = torch.abs(torch.randn((input_ids.shape[0], input_ids.shape[1], vocab_size), device=input_ids.device))
    # 对随机数进行归一化，使它们的和为x
    dis *= 1 - one_hot
    dis = torch.mul(F.normalize(dis, dim=-1, p=1), 1 - main_score)
    dis += torch.mul(one_hot, main_score)
    return dis


def generate_distribution3(input_ids, vocab_size, min_main_score=0.7):
    one_hot = F.one_hot(input_ids, vocab_size)
    main_score = (torch.randn(*one_hot.shape[:-1], device=input_ids.device) * 0.1 + min_main_score).unsqueeze(-1)
    main_score[((main_score > 1) | (main_score <= 0))] = min_main_score
    # 生成符合正态分布的随机数
    dis = torch.abs(torch.randn((input_ids.shape[0], input_ids.shape[1], vocab_size), device=input_ids.device))
    # 对随机数进行归一化，使它们的和为x
    dis *= 1 - one_hot
    dis = torch.mul(F.normalize(dis, dim=-1, p=1), 1 - main_score)
    dis += torch.mul(one_hot, main_score)
    return dis


def generate_spherical_vector(size: tuple, r: torch.Tensor = torch.tensor(1)):
    # 生成均匀分布在半径为 r 的超球面上的向量
    v = torch.randn(size, device=r.device)
    v = F.normalize(v, p=2, dim=-1)
    return v.mul(r)


def generate_ball_vector(size: tuple, r: torch.Tensor = torch.tensor(1)):
    # 生成均匀分布在半径为 r 的超球体内的向量
    size = size[:-1] + (size[-1] + 2,)
    v = torch.randn(size, device=r.device)
    v = F.normalize(v, p=2, dim=-1)
    return v.mul(r)[..., :-2]


def shift_embeddings(input_embs: torch.Tensor, alpha: float = 0.2) -> torch.Tensor:
    # 为每个 token 的 embedding 添加随机偏移, 平移量为原 embedding 向量的模的 alpha 倍
    r = torch.sqrt(input_embs.mul(input_embs).sum(dim=-1, keepdim=True)) * alpha
    return input_embs + generate_spherical_vector(input_embs.size(), r)

def shift_embeddings_ball(input_embs: torch.Tensor, alpha: float = 0.2) -> torch.Tensor:
    # 为每个 token 的 embedding 添加随机偏移, 平移量为原 embedding 向量的模的 alpha 倍
    r = torch.sqrt(input_embs.mul(input_embs).sum(dim=-1, keepdim=True)) * alpha
    return input_embs + generate_ball_vector(input_embs.size(), r)

# --------------------------------------------------------------------------------------------------------------


# TODO
def generate_distribution_matrix_wise(input_ids, vocab_size, hidden_size, min_main_score=0.7):
    one_hot = F.one_hot(input_ids, num_classes=vocab_size)
    one_hot = one_hot.unsqueeze(2).repeat_interleave(hidden_size, dim=2)
    main_score = (torch.rand(*one_hot.shape[:-1], dtype=torch.float32, device=input_ids.device) * (1 - min_main_score) + min_main_score).unsqueeze(-1)
    # # 生成符合正态分布的随机数
    dis = torch.abs(torch.randn((input_ids.shape[0], input_ids.shape[1], 768, vocab_size), device=input_ids.device))
    # # 对随机数进行归一化，使它们的和为x
    dis *= 1 - one_hot
    dis = F.normalize(dis, dim=-1, p=1) * (1 - main_score)
    dis += one_hot * main_score
    return dis
