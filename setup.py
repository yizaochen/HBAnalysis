from setuptools import setup, find_packages

setup(name='hbanalysis', 
      version='0.1',
      packages=find_packages(),
      url='https://github.com/yizaochen/HBAnalysis.git',
      author='Yizao Chen',
      author_email='yizaochen@gmail.com',
      install_requires=[
          'MDAnalysis',
          'matplotlib',
          'pandas',
          'scipy',
          'h5py'
      ]
      )