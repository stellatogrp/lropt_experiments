
import os

mypath = "./results/portfolio_results"
if not os.path.isdir(mypath):
   os.makedirs(mypath)

for i in range(16):
  mypath = f"./results/portfolio_results/results{i}"
  if not os.path.isdir(mypath):
    os.makedirs(mypath)