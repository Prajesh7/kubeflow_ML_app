import setuptools

with open("C:/Prajesh/personal/ML_learning/my_work/kubeflow_app/kubeflow_ML_app/README.md", 'r', encoding = 'utf-8') as f:
    long_description = f.read()
    
__version__ = "0.0.0"

REPO_NAME = "kubeflow_ML_app"
AUTHOR_USER_NAME = "Prajesh7"
SRC_REPO = "mlFlowProject"
AUTHOR_EMAIL = "prajeshlearningml@gmail.com"

setuptools.setup(
    name = SRC_REPO,
    version = __version__,
    author = AUTHOR_USER_NAME,
    author_email = AUTHOR_EMAIL,
    description = "A simple Python ML app",
    long_description = long_description,
    long_description_content = "text/markdown",
    url = f"https://github.com/{AUTHOR_USER_NAME}/{REPO_NAME}",
    project_urls = {
        "Bug Tracker": f"https://github.com/{AUTHOR_USER_NAME}/{REPO_NAME}/issues",
    },
    package_dir = {"": "src"},
    packages = setuptools.find_packages(where = "src")
)