from setuptools import setup

setup(
    name='circuit-transformer',
    version='1.2',
    packages=['circuit_transformer'],
    url='https://github.com/snowkylin/circuit-transformer',
    license='MIT',
    author='Xihan Li (snowkylin)',
    author_email='xihan.li@cs.ucl.ac.uk',
    description='An end-to-end Transformer model that efficiently produces logic circuits strictly equivalent to given Boolean functions.',
    python_requires='>=3.10',
    install_requires=["torch>=2.4", "numpy>=1.26", "scipy>=1.11", "npn",
                      "graphviz", "nvidia-ml-py", "bitarray", "huggingface_hub"],
    extras_require={"conversion": ["tensorflow>=2.18"], "test": ["pytest>=8"]},
    package_data={'circuit_transformer': ['bin/*']}
)
