import importlib
import subprocess
import sys

required_packages = [
    'numpy',
    'pandas',
    'scikit-learn',
    'torch',
    'shap',
    'matplotlib',
    'joblib',
    'tqdm',
    'pymongo',
    'python-dotenv',
]

optional_packages = [
    'ipywidgets',
    'ipython',
    'seaborn',
]

def check_package(package_name):
    """패키지가 설치되어 있는지 확인"""
    try:
        importlib.import_module(package_name)
        return True
    except ImportError:
        return False

def install_package(package_name):
    """패키지 설치 시도"""
    try:
        subprocess.check_call([sys.executable, '-m', 'pip', 'install', package_name])
        return True
    except subprocess.CalledProcessError:
        return False

def main():
    # 필수 패키지 확인
    missing_packages = []
    for package in required_packages:
        if not check_package(package):
            missing_packages.append(package)
    
    # 선택 패키지 확인
    missing_optional = []
    for package in optional_packages:
        if not check_package(package):
            missing_optional.append(package)
    
    if missing_packages:
        print("\n아래 필수 패키지가 설치되어 있지 않습니다:")
        for pkg in missing_packages:
            print(f"  - {pkg}")
        
        install = input("\n필요한 패키지를 설치하시겠습니까? (y/n): ")
        if install.lower() == 'y':
            for pkg in missing_packages:
                print(f"{pkg} 설치 중...")
                if install_package(pkg):
                    print(f"  {pkg} 설치 성공")
                else:
                    print(f"  {pkg} 설치 실패. 수동으로 설치하세요: pip install {pkg}")
    else:
        print("\n모든 필수 패키지가 설치되어 있습니다.")
    
    if missing_optional:
        print("\n아래 선택적 패키지가 설치되어 있지 않습니다:")
        for pkg in missing_optional:
            print(f"  - {pkg}")
        
        install = input("\n선택적 패키지를 설치하시겠습니까? (y/n): ")
        if install.lower() == 'y':
            for pkg in missing_optional:
                print(f"{pkg} 설치 중...")
                if install_package(pkg):
                    print(f"  {pkg} 설치 성공")
                else:
                    print(f"  {pkg} 설치 실패. 수동으로 설치하세요: pip install {pkg}")

    # torch GPU 확인
    if check_package('torch'):
        import torch
        print(f"\nPyTorch 버전: {torch.__version__}")
        print(f"CUDA 사용 가능: {torch.cuda.is_available()}")
        print(f"CUDA 버전: {torch.version.cuda}")
        
        # NVIDIA 드라이버 정보 확인
        try:
            import subprocess
            print("\nNVIDIA-SMI 정보:")
            result = subprocess.run(['nvidia-smi'], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            print(result.stdout.decode('utf-8'))
        except:
            print("nvidia-smi를 실행할 수 없습니다.")
        
        if torch.cuda.is_available():
            print("\nGPU 사용 가능:")
            for i in range(torch.cuda.device_count()):
                print(f"  - {torch.cuda.get_device_name(i)}")
        else:
            print("\nGPU를 사용할 수 없습니다. CPU만 사용됩니다.")
            print("\nCUDA 설치 방법:")
            print("  1. NVIDIA 드라이버 확인: nvidia-smi")
            print("  2. CUDA 재설치: pip uninstall torch && pip install torch torchvision torchaudio -f https://download.pytorch.org/whl/cu111/torch_stable.html")

if __name__ == "__main__":
    main()
