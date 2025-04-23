import setuptools

with open("requirements.txt", "r") as f:
    requirements = f.read().splitlines()

setuptools.setup(
    name="image-interpolation",
    version="0.1.0",
    author="BazeBai",
    author_email="author@example.com",
    description="Image Interpolation Prediction using ML, DL and Naive methods",
    long_description="""
    This project implements multiple methods for image interpolation:
    - Machine Learning (Random Forest) based interpolation
    - Deep Learning (UNet) based interpolation
    - Naive average-based interpolation
    
    It includes a Streamlit web interface for comparing these methods and 
    evaluating them using SSIM and PSNR metrics.
    """,
    long_description_content_type="text/markdown",
    url="https://github.com/BazeBai/image-interpolation",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
) 