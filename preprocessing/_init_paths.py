import os
import sys

if os.path.basename(os.path.abspath(os.getcwd())) == "preprocessing":
    os.chdir("..")

root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.join(root_dir))

