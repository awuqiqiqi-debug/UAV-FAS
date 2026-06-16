@echo off
echo ========================================
echo 安装 Twin-TD3 项目依赖库
echo ========================================

echo.
echo [1/3] 更新 pip...
python -m pip install --upgrade pip

echo.
echo [2/3] 安装核心依赖...
pip install -r requirements.txt

echo.
echo [3/3] 验证安装...
python -c "import torch; print(f'PyTorch: {torch.__version__}')"
python -c "import torchvision; print(f'torchvision: {torchvision.__version__}')"
python -c "import numpy; print(f'NumPy: {numpy.__version__}')"
python -c "import scipy; print(f'SciPy: {scipy.__version__}')"
python -c "import matplotlib; print(f'Matplotlib: {matplotlib.__version__}')"
python -c "import pandas; print(f'Pandas: {pandas.__version__}')"
python -c "import sklearn; print(f'Scikit-learn: {sklearn.__version__}')"
python -c "import seaborn; print(f'Seaborn: {seaborn.__version__}')"
python -c "import openpyxl; print(f'OpenPyXL: {openpyxl.__version__}')"
python -c "import pptx; print(f'python-pptx: {pptx.__version__}')"

echo.
echo ========================================
echo 安装完成！
echo ========================================
pause