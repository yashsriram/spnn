
from random import shuffle
with open('iris_data.txt') as f:
    content = f.readlines()
shuffle(content)
with open('iris_data.txt','w') as f:
    for line in content:
        f.write(line)

