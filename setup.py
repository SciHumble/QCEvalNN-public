from setuptools import setup, find_packages

setup(
    name="QCEvalNN",
    version="1.2.0",
    description="This package is for my master thesis about quantum neural "
                "networks and classical neural networks. "
                "The quantum simulation will be done with Pennylane. "
                "The classical portion with Pytorch.",
    author="Konrad Schubert",
    author_email="konrad.schubert@protonmail.com",
    url="https://github.com/SciHumble/QCEvalNN.git",
    keywords=["QNN", "QCNN", "bQCNN", "Qiskit", "spQCNN"],
    packages=find_packages(),
    install_requires=[
        "numpy", "matplotlib", "pylatexenc", "pytest", "scikit-learn",
        "IPython", "torch", "ptflops",
        "tensorflow", "pennylane", "pushbullet.py", "pandas", "torch",
        "tqdm", "scipy", "autograd", "requests"
        # Add other dependencies here
    ],
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Physics",
    ],
    python_requires='>=3.9',
    entry_points={
        'console_scripts': [
            # 'your_script_name = your_module:main_function',
        ],
    },
    license="MIT",
)
