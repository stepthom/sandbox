from nltk import Tree
from nltk.draw.util import CanvasFrame
from nltk.draw import TreeWidget

str0 = '[S [NP this tree] [VP [V is] [AdjP pretty]]]'


str1 = '[S  [NP [DT The] [NNS performances]] [VP [VBD were] [RB all] [ADJP [RB really] [JJ fantastic]]]]'

cf = CanvasFrame[]
t = Tree.fromstring[str1]
tc = TreeWidget[cf.canvas[],t]
cf.add_widget[tc,10,10] # [10,10] offsets
cf.print_to_file['tree.ps']
cf.destroy[]