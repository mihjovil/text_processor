import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name='package_name',
    version='1.0.0',
    author='Miguel Caldas',
    author_email='ma.caldas331@gmail.com',
    description='Text processor used in topic modelling project',
    long_description=long_description,
    long_description_content_type="text/markdown",
    url='https://github.com/mihjovil/text_processor',
    project_urls={
        "Other repos": "https://github.com/mihjovil?tab=repositories"
    },
    license='MIT',
    packages=['text_processor'],
    install_requires=['spacy', 'gensim', 'langdetect', 'pickle'],
)
