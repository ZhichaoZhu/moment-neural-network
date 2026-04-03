from pathlib import Path
import re

from setuptools import find_packages, setup


ROOT = Path(__file__).parent.resolve()
README = ROOT / "README.md"
PACKAGE_INIT = ROOT / "mnn" / "__init__.py"


def read_version() -> str:
    match = re.search(
        r'^__version__\s*=\s*["\']([^"\']+)["\']',
        PACKAGE_INIT.read_text(encoding="utf-8"),
        re.MULTILINE,
    )
    if not match:
        raise RuntimeError("Unable to find __version__ in mnn/__init__.py")
    return match.group(1)


setup(
    name="moment-neural-network",
    version=read_version(),
    description="A PyTorch-based framework for building and training moment neural networks.",
    long_description=README.read_text(encoding="utf-8"),
    long_description_content_type="text/markdown",
    author="Zhichao Zhu, Yang Qi",
    url="https://github.com/ZhichaoZhu/moment-neural-network",
    project_urls={
        "Source": "https://github.com/ZhichaoZhu/moment-neural-network",
        "Tracker": "https://github.com/ZhichaoZhu/moment-neural-network/issues",
        "Documentation": "https://github.com/ZhichaoZhu/moment-neural-network#readme",
    },
    license="Apache-2.0",
    license_files=("LICENSE",),
    packages=find_packages(
        exclude=("docs", "docs.*", "publications", "publications.*")
    ),
    include_package_data=True,
    python_requires=">=3.8",
    install_requires=[
        "numpy>=1.22",
        "scipy>=1.7",
        "PyYAML>=6.0",
        "tqdm>=4.0",
        "torch>=1.12",
        "torchvision>=0.13",
    ],
    extras_require={
        "analysis": [
            "matplotlib>=3.5",
            "pandas>=1.4",
            "seaborn>=0.11",
        ],
        "loihi": [
            "attrdict",
            "h5py",
            "matplotlib>=3.5",
            "pandas>=1.4",
            "seaborn>=0.11",
        ],
    },
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3 :: Only",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    keywords=[
        "moment neural network",
        "spiking neural network",
        "machine learning",
        "deep learning",
        "pytorch",
    ],
)
