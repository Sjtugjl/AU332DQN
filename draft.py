from sumtree import SumTree




tree = SumTree(memory_size=10)
p = 1
for i in range(p):
    tree.add(10000, (1, 1, 1, 1, 1))
print("tree",tree.tree)
print("transition",tree.transitions)
