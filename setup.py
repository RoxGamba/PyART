from setuptools import setup, find_packages

setup(
    name='PyART',
    version='0.0.1',    
    description='Python Analytcial Relativity Toolkit',
    url='https://github.com/RoxGamba/PyART',
    author='The TEOBResumS team',
    author_email='rgamba@berkeley.edu',
    license='MIT',
    packages= find_packages(),
    install_requires=[
                      'numpy',                     
                      ],

    classifiers=[
        'Development Status :: 1 - Planning',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: BSD License',  
        'Programming Language :: Python :: 3.8',
    ],
)
