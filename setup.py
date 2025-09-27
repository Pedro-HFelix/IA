from setuptools import setup, find_packages

setup(
    name='trees_classifiers', 
    
    version='0.1.0',
    
    author='Felix',
    author_email='pf71136@gmail.com',
    
    description='lista 05',
    
    packages=find_packages(),
    
    install_requires=[
        'numpy',
        'pandas',
        'scikit-learn' 
    ],
    
    keywords=['decision tree', 'id3', 'c45', 'cart',],
    
    python_requires='>=3.8',
)