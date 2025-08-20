from setuptools import setup, find_packages

setup(
    name='openrct2_gym',
    version='0.1.0',
    packages=find_packages(),
    install_requires=[
        'gymnasium>=0.29.0',
        'numpy>=1.24.0',
    ],
    extras_require={
        'train': [
            'stable-baselines3>=2.0.0,<2.8.0',
            'sb3-contrib>=2.0.0,<2.8.0',
            'tensorboard>=2.13.0',
        ],
        'dev': [
            'pytest>=7.4.0',
            'black>=23.0.0',
            'flake8>=6.0.0',
        ]
    },
    python_requires='>=3.8',
    description='OpenRCT2 Gymnasium environment for training RL agents to build roller coasters',
    author='Your Name',
    license='MIT',
)

