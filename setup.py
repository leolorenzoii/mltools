import os
import sys
import site
import subprocess


def install(package):
    """Install package using pip"""
    subprocess.check_call([sys.executable, "-m", "pip", "install", package])

# Get site-packages directory
site_packages_dir = site.USER_SITE

# Create site-packages directory if not exists
if not os.path.exists(site_packages_dir):
    print("Creating site packages directory..")
    os.makedirs(site_packages_dir)

# Get current directory
cur_dir = os.path.dirname(__file__)
cur_dir = os.path.abspath(cur_dir)

# Set skils path file
mltools_pth = site_packages_dir + '/mltools.pth'

# Add to site-packages dir the path of skils
with open(mltools_pth, 'w') as f:
    f.write(cur_dir)
print(f"Added {mltools_pth} on {site_packages_dir}") 

# Install required packages
print("Installing required packages:\n")
with open('requirements.txt', 'r') as f:
    for package in f.readlines():
        install(package)
