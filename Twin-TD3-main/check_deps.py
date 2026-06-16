#!/usr/bin/env python3
"""
检查所有依赖库是否正确安装
"""
import sys

def check_package(package_name, import_name=None):
    """检查包是否已安装"""
    if import_name is None:
        import_name = package_name
    try:
        module = __import__(import_name)
        version = getattr(module, '__version__', '未知版本')
        print(f"[OK] {package_name}: {version}")
        return True
    except ImportError:
        print(f"[FAIL] {package_name}: 未安装")
        return False

def main():
    print("=" * 50)
    print("Twin-TD3 项目依赖检查")
    print("=" * 50)
    print()

    packages = [
        # 核心科学计算
        ("numpy", "numpy"),
        ("scipy", "scipy"),
        ("pandas", "pandas"),

        # 机器学习/深度学习
        ("torch", "torch"),
        ("torchvision", "torchvision"),
        ("scikit-learn", "sklearn"),

        # 可视化
        ("matplotlib", "matplotlib"),
        ("seaborn", "seaborn"),
        ("plotly", "plotly"),

        # 数据处理
        ("openpyxl", "openpyxl"),
        ("python-dateutil", "dateutil"),
        ("pytz", "pytz"),

        # PPT生成
        ("python-pptx", "pptx"),

        # 工具库
        ("joblib", "joblib"),
        ("pyparsing", "pyparsing"),
    ]

    success_count = 0
    total_count = len(packages)

    for package_name, import_name in packages:
        if check_package(package_name, import_name):
            success_count += 1

    print()
    print("=" * 50)
    print(f"检查结果: {success_count}/{total_count} 个包已安装")

    if success_count == total_count:
        print("[OK] 所有依赖库已正确安装！")
        return 0
    else:
        print("[FAIL] 部分依赖库缺失，请运行 install_deps.bat 安装")
        return 1

if __name__ == "__main__":
    sys.exit(main())
