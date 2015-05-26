import orange
import Orange
import orngTree
import itertools


debug_data = orange.ExampleTable('C:\Python27\Lib\site-packages\Orange\datasets\\titanic.tab')
debug_data = orange.ExampleTable('C:\Python27\Lib\site-packages\Orange\datasets\lung-cancer.tab')

debug_tree = orngTree.TreeLearner(debug_data)
debug_rforest = Orange.ensemble.forest.RandomForestLearner(debug_data)
debug_major = orange.MajorityLearner(debug_data)

debug_bayes = orange.BayesLearner(debug_data)
debug_bayes.setattr("data", debug_data)

cl = debug_tree

domain = cl.domain

# New sample data
combinations = []
combinations_count = 1;
for attr in domain.attributes:
    values = attr.values
    combinations.append(values)
    combinations_count = combinations_count * len(values)


#data.domain.attributes[0].values
#Orange.data.Table
#new_data = data.clone() 
new_data = Orange.data.Table(domain, [])
for combination in itertools.product(*combinations):
    combination_list = list(combination)
    combination_list.append('?')
    di = Orange.data.Instance(domain, combination_list)
    c = cl(di)
    combination_list.pop()
    combination_list.append(c)
    new_data.append(combination_list)
    
bayes = orange.BayesLearner(new_data)

for item in new_data:
    result = item[3]
    item[3] = '?'
    print str(item) + ' ' + str(result) + ' = '+ str(bayes(item))