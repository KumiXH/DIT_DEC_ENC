from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor


def latent_smooth_l1(student: Tensor, teacher: Tensor) -> Tensor:
    return F.smooth_l1_loss(student, teacher)


def cosine_loss(student: Tensor, teacher: Tensor) -> Tensor:
    if student.shape != teacher.shape:
        raise ValueError(f"cosine tensors must have equal shape, got {student.shape} and {teacher.shape}")
    student_vectors = student.movedim(1, -1).reshape(-1, student.shape[1])
    teacher_vectors = teacher.movedim(1, -1).reshape(-1, teacher.shape[1])
    return (1.0 - F.cosine_similarity(student_vectors, teacher_vectors, dim=-1)).mean()


def channel_stat_loss(student: Tensor, teacher: Tensor) -> Tensor:
    reduce_dims = (0, *range(2, student.ndim))
    student_mean = student.mean(dim=reduce_dims)
    teacher_mean = teacher.mean(dim=reduce_dims)
    student_std = student.std(dim=reduce_dims, unbiased=False)
    teacher_std = teacher.std(dim=reduce_dims, unbiased=False)
    return F.l1_loss(student_mean, teacher_mean) + F.l1_loss(student_std, teacher_std)


def edge_loss(student: Tensor, target: Tensor) -> Tensor:
    def gradients(image: Tensor) -> tuple[Tensor, Tensor]:
        horizontal = image[..., :, 1:] - image[..., :, :-1]
        vertical = image[..., 1:, :] - image[..., :-1, :]
        return horizontal, vertical

    student_h, student_v = gradients(student)
    target_h, target_v = gradients(target)
    return F.l1_loss(student_h, target_h) + F.l1_loss(student_v, target_v)

