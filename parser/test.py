from __future__ import print_function, unicode_literals
import Smipar, json

test_string1 = '[c:7]1([CH3:6])[c:12]([C:3]([c:2]2[cH:11][cH:12][cH:7][cH:8][c:9]2[CH3:10])=[O:5])[cH:11][cH:10][cH:9][cH:8]1'
test_string2 = 'CN1CCC[C@H:3]1(c2cc)c[*]nc2'

# parse to json test
print(('---TEST1---'))
print((Smipar.parser_json(test_string1)))

# parse to list test
print(('---TEST2---'))
print((Smipar.parser_list(test_string2)))
print('test smiles string:', test_string2)
print('parsed-joined result:', ''.join(Smipar.parser_list(test_string2)))
print('(should be same)')
