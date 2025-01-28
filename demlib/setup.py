from setuptools import setup, find_packages

setup(
    name='demlib',  # The name of your package
    version='0.1',  # Version of your package
    packages=find_packages(where='demlib'),  # Look for packages inside the 'demlib' folder
    package_dir={'': 'demlib'},  # Tells setuptools to look inside 'demlib' for Python packages
    install_requires=[  # List of dependencies
        'scipy',  # Include any other dependencies that your package needs
        'numpy',
    ],
    # Metadata
    author='Gerard Batet',
    author_email='gerard.batet@upc.edu',
    description='A library for demodulation PPM and ADC emulation.',
    long_description=open('README.md').read(),  # You can add a README for long description
    long_description_content_type='text/markdown',
    url='https://github.com/b3rax/PPM-ULP-algorithms/demlib',  # Replace with your package's URL
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: CC0',  # Specify the license type
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.12',  # Specify the minimum Python version
)