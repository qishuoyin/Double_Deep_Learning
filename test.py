import os
from pathlib import Path
  
code_location = os.path.dirname(__file__)
dataset_location = code_location + '/data_simulation'
print(code_location)
print(dataset_location)

parent_location = Path(__file__) #.parent
print(parent_location)
