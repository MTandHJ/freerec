import setuptools

with open("README.md", "r") as fh:
  long_description = fh.read()

requires = [
    "torchdata>=0.4.0",
    "torchmetrics>=0.9.2",
    "feather-format>=0.4.1",
    "tqdm>=4.64.0",
    "freeplot>=0.0.12"
]

setuptools.setup(
  name="freerec",
  version="0.0.5",
  author="MTandHJ",
  author_email="congxueric@gmail.com",
  description="PyTorch library for recommender systems",
  long_description=long_description,
  long_description_content_type="text/markdown",
  license='MIT License',
  url="https://github.com/MTandHJ/freerec",
  packages=setuptools.find_packages(),
  python_requires='>=3.7',
  install_requires=requires,
  classifiers=[
  "Programming Language :: Python :: 3",
  "License :: OSI Approved :: MIT License",
  "Operating System :: OS Independent",
  ],
)