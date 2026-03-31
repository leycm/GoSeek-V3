# Python Interface for Go-based DeepSeek Model Conversion
#
# This file is a Python Interface running the go-based impl
# to support same usage.
#
# Written by Lennard <leycm@proton.me>, 2026.
#
# This project is not affiliated with or endorsed by DeepSeek.
import sys
import subprocess
args = sys.argv[1:]

command = ["go", "run", "../go/fp8_cast_bf16.go"] + args
process = subprocess.Popen(
    command,
    stdin=sys.stdin,
    stdout=sys.stdout,
    stderr=sys.stderr
)

 process.wait()
 sys.exit(process.returncode)