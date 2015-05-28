import orange
import Orange
import orngTree
import itertools
from xml.dom import WRONG_DOCUMENT_ERR


debug_data = orange.ExampleTable('C:\Python27\Lib\site-packages\Orange\datasets\\titanic.tab')
debug_data = orange.ExampleTable('C:\Python27\Lib\site-packages\Orange\datasets\lung-cancer.tab')

debug_tree = orngTree.TreeLearner(debug_data)
debug_rforest = Orange.ensemble.forest.RandomForestLearner(debug_data)
debug_major = orange.MajorityLearner(debug_data)

debug_bayes = orange.BayesLearner(debug_data)
debug_bayes.setattr("data", debug_data)

cl = debug_bayes

org_domain = cl.domain

domain = Orange.data.Domain(org_domain.features[3:10] + [org_domain.class_var])

# New sample data
combinations = []
combinations_count = 1
attr_count = 0
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
lr = Orange.classification.logreg.LogRegLearner(new_data)


print domain
bwrong = 0
right =  0
lrwrong = 0
lrright =  0
for item in new_data:
    item = list(item)
    result = item.pop()
    item.append('?')
    di = Orange.data.Instance(domain, item)
    b_cls = bayes(di)
    lr_cls = lr(di)
    if str(result) != str(b_cls):
        #print str(item) + ' ' + str(result) + ' = '+ str(cls)
        bwrong += 1
    else:
        bright += 1
    if str(result) != str(lr_cls):
        #print str(item) + ' ' + str(result) + ' = '+ str(cls)
        lrwrong += 1
    else:
        lrright += 1
print 'Bayes Accurate: ' + str(100 - bwrong*1.0/(bwrong+bright)*100.0) +'%'
print 'LR Accurate: ' + str(100 - lrwrong*1.0/(lrwrong+lrright)*100.0) +'%'