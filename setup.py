import setuptools
import re

with open("README.md", "r") as fh:
  long_description = fh.read()

requires = [
    # "torchdata==0.4.1",
    # "freeplot>=0.4.5",
    "polars>=1.9.0",
    "PyYAML>=6.0",
    "tensorboard>=2.10.0",
    "prettytable>=3.4.1"
]


def get_property(prop, project):
    result = re.search(r'{}\s*=\s*[\'"]([^\'"]*)[\'"]'.format(prop), open(project + '/__init__.py').read())
    return result.group(1)


setuptools.setup(
  name="freerec",
  entry_points={
    'console_scripts': [
      'freerec=freerec.__main__:main',
    ],
  },
  version=get_property('__version__', 'freerec'),
  author="MTandHJ",
  author_email="congxueric@gmail.com",
  description="PyTorch library for recommender systems",
  long_description=long_description,
  long_description_content_type="text/markdown",
  license='MIT License',
  url="https://github.com/MTandHJ/freerec",
  package_data={
      "freerec": [
          "data/postprocessing/*.pyi",
      ],
  },
  packages=setuptools.find_packages(),
  python_requires='>=3.9',
  install_requires=requires,
  classifiers=[
  "Programming Language :: Python :: 3",
  "License :: OSI Approved :: MIT License",
  "Operating System :: OS Independent",
  ],
)