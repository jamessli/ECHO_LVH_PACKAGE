from setuptools import setup, find_packages

setup(
    name='ECHO_LVH_PACKAGE',
    install_requires=['pandas','scikit-learn', 'tensorflow', 'numpy', 'matplotlib.cm', 'cv2', 'natsort', 'pathlib', 'isort', 'streamlit', 'IPython.display', 'pickle', 'time'],
    packages=find_packages()
)
