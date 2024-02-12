import pdb
import subprocess

ROOTD = subprocess.check_output("git rev-parse --show-toplevel", shell=True).strip().decode("utf-8")
