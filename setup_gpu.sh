#!/bin/bash

echo "GTX1080 GPU 설정 스크립트 시작..."

# CUDA 버전 확인
nvidia-smi

# 기존 PyTorch 제거
pip uninstall -y torch torchvision torchaudio

# CUDA 11.1 호환 PyTorch 설치
pip install torch torchvision torchaudio -f https://download.pytorch.org/whl/cu111/torch_stable.html

# 설치 확인
python -c "import torch; print('CUDA 사용 가능:', torch.cuda.is_available()); print('CUDA 버전:', torch.version.cuda); print('GPU 이름:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'None')"

echo "설정 완료"
