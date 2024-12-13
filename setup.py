from setuptools import setup

with open('README.md') as f:
    readme = f.read()

with open('requirements.txt') as fp:
    install_requires = fp.read()

setup(name='OzoneRates',
      version='1.0.0',
      description='Global estimates of PO3 from space',
      long_description=readme,
      long_description_content_type='text/markdown',
      author=['Amir Souri', 'Gonzalo Gonzalez Abad'],
      author_email='ahsouri@gmail.com',
      license='MIT',
      packages=['ozonerates'],
      install_requires=install_requires,
      zip_safe=False)
