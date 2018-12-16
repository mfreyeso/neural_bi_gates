from setuptools import setup, find_packages

setup(name='neural_gates',
      version='1.0',
      packages=find_packages(),
      include_package_data=True,
      description='Neural Network for a Logical Binary Operation',
      author='Mario Reyes Ojeda',
      author_email='mfreyesojeda@gmail.com',
      install_requires=[
          'keras',
          'h5py'],
      zip_safe=False)