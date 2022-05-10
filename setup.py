import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name='package_name',
    version='0.0.1',
    author='Miguel Caldas',
    author_email='ma.caldas331@gmail.com',
    description='Template for installation of Package',
    long_description=long_description,
    long_description_content_type="text/markdown",
    url='https://github.com/mihjovil/package_template',
    project_urls={
        "Other repos": "https://github.com/mihjovil?tab=repositories"
    },
    license='MIT',
    packages=['package_name'],
    install_requires=['requests'],
)
