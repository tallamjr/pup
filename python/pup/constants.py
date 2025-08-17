import subprocess

ROOTD = subprocess.check_output(["git", "rev-parse", "--show-toplevel"]).strip().decode("utf-8")
