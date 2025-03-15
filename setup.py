from setuptools import setup, find_packages

setup(
    name='fancyclusters',
    version='0.1.0',
    packages=find_packages(),
    install_requires=[
        'pandas',
        'numpy',
        'scikit-learn'
    ],
    author = 'dabe19',
    author_email='djmccabe87@gmail.com',
    description='A package for enhanced clustering with pandas and numpy integration.',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    url='https://github.com/dabe-19/fancyclusters',
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.10',
)