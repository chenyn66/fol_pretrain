
import random


class Node:
    def __init__(self, name) -> None:
        self.truth = random.choice([True, False])
        self.name = name
        self.children = []
        self.nots = []
        self.ops = []
        self.generate_children()
        
    def generate_children(self):
        num = random.randint(0, 2)
        if num >= 1:
            self.children.append(Node(self.name + '0'))
            self.nots.append(random.choice([True, False]))
        if num >= 2:
            self.children.append(Node(self.name + '1'))
            self.nots.append(random.choice([True, False]))
            self.ops.append('and')
            
            
    def evaluate(self):
        if len(self.children) == 1:
            self.truth =  self.children[0].evaluate() == self.nots[0]
        elif len(self.children) == 2:
            if self.nots[0] == 0:
                a = self.children[0].evaluate()
            else:
                a = self.children[0].evaluate() == False
            if self.nots[0] == 0:
                b = self.children[0].evaluate()
            else:
                b = self.children[0].evaluate() == False
            if self.ops[0] == 'and':
                self.truth =  a == b == True
        return self.truth
    
    def print_statements(self):
        for child in self.children:
            child.print_statements()
            print("")
        if len(self.children) == 0:
            print('{} = {}'.format(self.name, self.truth))
        elif len(self.children) == 1:
            print('{} = {} -> {}'.format(self.children[0].name, self.nots[0], self.name))
        elif len(self.children) == 2:
            print('{} = {} {} {} = {} -> {}'.format(self.children[0].name, self.nots[0], self.ops[0], self.children[1].name, self.nots[1], self.name))
            
            
            
root = Node('0')
root.evaluate()
root.print_statements()
print("")
print("{} is {}".format(root.name, root.truth))
