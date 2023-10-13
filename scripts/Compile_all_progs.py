import os.path
import subprocess
from pathlib import Path


full_path_to_rga = Path(os.path.join(os.getcwd(), "./rga.exe"))
full_path_to_rga_temp = Path("C:/Users/Denis/Downloads/RadeonDeveloperToolSuite-2023-09-18-1222/RadeonDeveloperToolSuite-2023-09-18-1222/rga.exe")

print("full_path_to_rga: ", full_path_to_rga)
test_folder = os.path.join(full_path_to_rga.parent.parent, "Tests")

#Simple add:
simpleAddFolder = os.path.join(test_folder, "SimpleAdd")

add_kernel = os.path.join(simpleAddFolder, "Add.cl")
print("add_kernel", add_kernel)

disasm = os.path.join(simpleAddFolder, "disassem.txt")
binary = os.path.join(simpleAddFolder, "prog.bin")
rga_command = [full_path_to_rga_temp, "-s", "opencl", "-c", "gfx1034", "--isa", disasm,  "-b", binary, "--O0", add_kernel]
print("rga_command", rga_command)
result = subprocess.call(rga_command);
print("Result: ", result)
