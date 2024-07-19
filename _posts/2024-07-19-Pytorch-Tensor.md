---
layout: single
title: "Learned Pytorch for the first time"
categories: "Machine Learning"
published: true
---

# Pytorch Tensor

# Pytorch Tensor 생성

```python
import torch
import numpy as np

list_data = [[10, 20], [30, 40]]

tensor1 = torch.Tensor(list_data)

# 아래 코드를 통해서 cpu에 올라간 데이터를 gpu로 옮길 수 있음
if torch.cuda.is_available():
  tensor1 = tensor1.to('cuda')

print(tensor1)
print(f'tensor type: {type(tensor1)}')
print(f'tensor shape: {tensor1.shape}')
print(f'tensor dtype: {tensor1.dtype}')
print(f'tensor device: {tensor1.device}')
```

    tensor([[10., 20.],
            [30., 40.]], device='cuda:0')
    tensor type: <class 'torch.Tensor'>
    tensor shape: torch.Size([2, 2])
    tensor dtype: torch.float32
    tensor device: cuda:0

# Numpy array를 Pytorch Tensor로 변환

```python
# Pytorch는 numpy array로 tensor를 만들 수 있음
# Deep Learning에서는 float형이 기본인데 numpy array로 만든 tensor는 int형으로 생성되므로 float형으로 type casting을 해줘야 함

numpy_data = np.array(list_data)

tensor2_1 = torch.from_numpy(numpy_data)

print(tensor2_1)
print(f'tensor type: {type(tensor2_1)}')
print(f'tensor shape: {tensor2_1.shape}')
print(f'tensor dtype: {tensor2_1.dtype}')
print(f'tensor device: {tensor2_1.device}')

print('============================')

tensor2_2 = torch.from_numpy(numpy_data).float()

print(tensor2_2)
print(f'tensor type: {type(tensor2_2)}')
print(f'tensor shape: {tensor2_2.shape}')
print(f'tensor dtype: {tensor2_2.dtype}')
print(f'tensor device: {tensor2_2.device}')
```

    tensor([[10, 20],
            [30, 40]])
    tensor type: <class 'torch.Tensor'>
    tensor shape: torch.Size([2, 2])
    tensor dtype: torch.int64
    tensor device: cpu
    ============================
    tensor([[10., 20.],
            [30., 40.]])
    tensor type: <class 'torch.Tensor'>
    tensor shape: torch.Size([2, 2])
    tensor dtype: torch.float32
    tensor device: cpu

# Pytorch rand() / randn() 메소드

```python
# rand(): 0에서 1 사이에서 균일한 분포의 random 값 생성
tensor3 = torch.rand(2, 2)
print(tensor3)

# randn(): 평균이 0이고 분산이 1인 정규분포를 갖는 random 값 생성 -> 딥러닝에서 weights와 bias 등을 초기화 할 때 자주 사용
tensor4 = torch.randn(2, 2)
print(tensor4)
```

    tensor([[0.6208, 0.4878],
            [0.2148, 0.7456]])
    tensor([[-1.9481,  0.1075],
            [ 0.7545, -0.8256]])

# Pytorch Tensor를 Numpy array로

```python
# pytorch Tensor는 소수점 이하 네 자리까지 표시
tensor5 = torch.randn(2, 2)
print(tensor5)

# numpy 배열은 pytorch Tensor보다 더 길게 표시
numpy_from_tensor = tensor5.numpy()
print(numpy_from_tensor)
```

    tensor([[-0.2667, -1.0358],
            [-1.1553, -0.5216]])
    [[-0.266718   -1.0358081 ]
     [-1.1553456  -0.52164817]]

# Pytorch Tensor의 indexing / slicing

```python
tensor6 = torch.Tensor([[1, 2, 3], [4, 5, 6]])
tensor7 = torch.Tensor([[7, 8, 9], [10, 11, 12]])

print(tensor6[0])
print(tensor6[:, 1:])
print(tensor7[0:2, 0:-1])
print(tensor7[-1, -1])
print(tensor7[..., -2])
```

    tensor([1., 2., 3.])
    tensor([[2., 3.],
            [5., 6.]])
    tensor([[ 7.,  8.],
            [10., 11.]])
    tensor(12.)
    tensor([ 8., 11.])

## Pytorch Tensor의 곱셈 연산

```python
# mul() 메소드는 tensor 요소 간 곱셈 계산
tensor8 = tensor6.mul(tensor7)
print(tensor8)

# matmul() 메소드는 내적(행렬 곱셈) 계산
# tensor9 = tensor6.matmul(tensor7) -> tensor6와 tensor7의 shape가 맞지 않기 때문에 Error 발생
tensor9 = tensor6.matmul(tensor7.view(3, 2))
print(tensor9)
```

    tensor([[ 7., 16., 27.],
            [40., 55., 72.]])
    tensor([[ 58.,  64.],
            [139., 154.]])

# Pytorch Tensor Concatenate

```python
tensor_cat = torch.cat([tensor6, tensor7])
print(tensor_cat)
```

    tensor([[ 1.,  2.,  3.],
            [ 4.,  5.,  6.],
            [ 7.,  8.,  9.],
            [10., 11., 12.]])

```python
# dim=0: Concatenation along rows
tensor_cat_dim0 = torch.cat([tensor6, tensor7], dim=0)
print(tensor_cat_dim0)
```

    tensor([[ 1.,  2.,  3.],
            [ 4.,  5.,  6.],
            [ 7.,  8.,  9.],
            [10., 11., 12.]])

```python
# dim=1: Concatenation along columns
tensor_cat_dim1 = torch.cat([tensor6, tensor7], dim=1)
print(tensor_cat_dim1)
```

    tensor([[ 1.,  2.,  3.,  7.,  8.,  9.],
            [ 4.,  5.,  6., 10., 11., 12.]])
