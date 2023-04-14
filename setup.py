import subprocess
import sys

from setuptools import setup, find_packages

with open("README.md", "r") as f:
    readme = f.read()

def check_if_conda_package_installed(package_name):
    result = subprocess.run(["conda", "list", package_name], stdout=subprocess.PIPE)
    output = result.stdout.decode("utf-8")
    return package_name in output

def get_url() -> str:
    if sys.version_info[:2] == (3, 8):
        _ver = "cp38"
    elif sys.version_info[:2] == (3, 9):
        _ver = "cp39"
    elif sys.version_info[:2] == (3, 10):
        _ver = "cp310"
    else:
        raise Exception(f"Python {sys.version} is not supported.")

    # FIXME: how to download on macox??
    if sys.platform == "win32":
        _os = "win_amd64"
    else:
        _os = "linux_x86_64"

    return f"https://download.pytorch.org/whl/cu113/torch-1.12.0%2Bcu113-{_ver}-{_ver}-{_os}.whl"

required_packages = [
    "biopython"
]

try:
    if not check_if_conda_package_installed("pytorch"):
        required_packages.append(f"torch@{get_url()}")
except:
    # e.g., if `conda` is not installed, use `pip` to ensure PyTorch CUDA is installed
    required_packages.append(f"torch@{get_url()}")

setup(
    name="OmegaFold",
    description="OmegaFold Release Code",
    long_description=readme,
    long_description_content_type="text/markdown",
    license="Apache-2.0",
    packages=find_packages(exclude=["tests", "tests.*"]),
    entry_points={"console_scripts": ["omegafold=omegafold.__main__:main",],},
    install_requires=required_packages,
    python_requires=">=3.8",
)
