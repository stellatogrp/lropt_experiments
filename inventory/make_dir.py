
import os

mypath = "./results/inventory_results"
if not os.path.isdir(mypath):
   os.makedirs(mypath)

for i in range(16):
  mypath = f"./results/inventory_results/results{i}"
  if not os.path.isdir(mypath):
    os.makedirs(mypath)