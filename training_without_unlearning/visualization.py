import matplotlib.pyplot as plt
from csv import reader
import numpy as np
import csv
import ast


datasets = ['compas']#,'compas','default_credit', 'marketing']
y = ['10%','20%', '30%','40%', '50%',   '60%', '70%','80%','90%','95%']
model = 'NN'

for dataset in datasets:
    lists = [[] for _ in range(11)]
    lists1 = [[] for _ in range(11)]

    summary = [[] for _ in range(11)]
    per  = 50
    file = './results/summary/summary_{}_{}.csv'.format(dataset,model)
    file2 = '../results/summary/summary_{}_{}.csv'.format(dataset,model)

    title = 'fairness result for {} trained using {}'.format(dataset,model)
#     title2 = 'fairness result for {} trained using {}'.format(dataset,model)
    csv.register_dialect('mydialect', delimiter = ',', quotechar = '"', doublequote = True, skipinitialspace = True, quoting = csv.QUOTE_MINIMAL)
    with open(file2, 'r') as read_obj:
    #         lines = reader(read_obj)
        lines = csv.reader(read_obj, dialect='mydialect')

        for row in lines:
            print(row)
            lists1[0].append(row[4]) #sp
            lists1[1].append(row[5])
            lists1[2].append(row[6]) #pe
            lists1[3].append(row[7])
            lists1[4].append(row[8]) #eopp
            lists1[5].append(row[9])
            lists1[6].append(row[10]) #eodd
            lists1[7].append(row[11])
            lists1[8].append(row[1]) #y
            lists1[9].append(row[2]) # acc
            lists1[10].append(row[3]) # std

#         print(lists)
    with open(file, 'r') as read_obj:
    #         lines = reader(read_obj)
        lines = csv.reader(read_obj, dialect='mydialect')

        for row in lines:
            lists[0].append(row[4]) #sp
            lists[1].append(row[5])
            lists[2].append(row[6]) #pe
            lists[3].append(row[7])
            lists[4].append(row[8]) #eopp
            lists[5].append(row[9])
            lists[6].append(row[10]) #eodd
            lists[7].append(row[11])
            lists[8].append(row[1]) #y
            lists[9].append(row[2]) # acc
            lists[10].append(row[3]) # std

#         print(lists)
    # Increase the width
    plt.figure(figsize=(10,4))
#     plt.grid(True, color = "grey", linewidth = "2", linestyle = "--")
    sp = [ast.literal_eval(i) for i in lists[0][1:]]
    err1 = [ast.literal_eval(i) for i in lists[1][1:]]
    sp2 = [ast.literal_eval(i) for i in lists1[0][1:]]
    err2 = [ast.literal_eval(i) for i in lists1[1][1:]]
    
    pe = [ast.literal_eval(i) for i in lists[2][1:]]
    err3 = [ast.literal_eval(i) for i in lists[3][1:]]
    pe2 = [ast.literal_eval(i) for i in lists1[2][1:]]
    err4 = [ast.literal_eval(i) for i in lists1[3][1:]]
    
    eopp = [ast.literal_eval(i) for i in lists[4][1:]]
    err5 = [ast.literal_eval(i) for i in lists[5][1:]]
    eopp2 = [ast.literal_eval(i) for i in lists1[4][1:]]
    err6 = [ast.literal_eval(i) for i in lists1[5][1:]]
    
    eodd = [ast.literal_eval(i) for i in lists[6][1:]]
    err7 = [ast.literal_eval(i) for i in lists[7][1:]]
    eodd2 = [ast.literal_eval(i) for i in lists1[6][1:]]
    err8 = [ast.literal_eval(i) for i in lists1[7][1:]]
    
    

    
    plt.plot(y, sp[1:],  color = 'g', linestyle = '-', label = "SP without sisa")
    plt.errorbar(y, sp[1:], yerr=err1[1:], fmt='go',capsize=5, elinewidth=2, markeredgewidth=2)
    plt.plot(y, sp2, color = 'b', linestyle = '--', label = "SP with sisa")
    plt.errorbar(y, sp2, yerr=err2,fmt='bo',capsize=10, elinewidth=2, markeredgewidth=2)
    plt.xlabel('Num of requests')
    plt.ylabel('unfairness')
    plt.legend(loc ='upper right')
    plt.title(title, fontsize = 20)
    plt.savefig('./results/summary/{}_{}_SP.png'.format( dataset,model))    
    plt.show() 
    
    plt.figure(figsize=(10,4))
    plt.plot(y, pe[1:],  color = 'g', linestyle = '-', label = "PE without sisa")
    plt.errorbar(y, pe[1:], yerr=err3[1:], fmt='go',capsize=5, elinewidth=2, markeredgewidth=2)
    plt.plot(y, pe2, color = 'b', linestyle = '--', label = "PE with sisa")
    plt.errorbar(y, pe2, yerr=err4,fmt='bo',capsize=10, elinewidth=2, markeredgewidth=2)
    plt.xlabel('Num of requests')
    plt.ylabel('unfairness')
    plt.legend(loc ='upper right')
    plt.title(title, fontsize = 20)
    plt.savefig('./results/summary/{}_{}_PE.png'.format( dataset,model))    
    plt.show() 
    
    plt.figure(figsize=(10,4))
    plt.plot(y, eopp[1:],  color = 'g', linestyle = '-', label = "EOpp without sisa")
    plt.errorbar(y, eopp[1:], yerr=err5[1:], fmt='go',capsize=5, elinewidth=2, markeredgewidth=2)
    plt.plot(y, eopp2, color = 'b', linestyle = '--', label = "EOpp with sisa")
    plt.errorbar(y, eopp2, yerr=err6, fmt='bo',capsize=15, elinewidth=3, markeredgewidth=2)
    plt.xlabel('Num of requests')
    plt.ylabel('unfairness')
    plt.legend(loc ='upper right')
    plt.title(title, fontsize = 20)
    plt.savefig('./results/summary/{}_{}_eopp.png'.format( dataset,model))    
    plt.show() 
    
    
    plt.figure(figsize=(10,4))
    plt.plot(y, eodd[1:],  color = 'g', linestyle = '-', label = "EOdds without sisa")
    plt.errorbar(y, eodd[1:], yerr=err7[1:], fmt='go',capsize=5, elinewidth=2, markeredgewidth=2)
    plt.plot(y, eodd2, color = 'b', linestyle = '--', label = "EOdds with sisa")
    plt.errorbar(y, eodd2, yerr=err8,fmt='bo',capsize=10, elinewidth=2, markeredgewidth=2)
    plt.xlabel('Num of requests')
    plt.ylabel('unfairness')
    plt.legend(loc ='upper right')
    plt.title(title, fontsize = 20)
    plt.savefig('./results/summary/{}_{}_eodd.png'.format( dataset,model))    
    plt.show() 
    
#     ################################### fairness for 80%###############
#     plt.figure(figsize=(10,6))
#     sp = [ast.literal_eval(i) for i in lists1[0][1:]]
#     err1 = [ast.literal_eval(i) for i in lists1[1][1:]]
#     err2 = [ast.literal_eval(i) for i in lists1[3][1:]]
#     err3 = [ast.literal_eval(i) for i in lists1[5][1:]]
#     err4 = [ast.literal_eval(i) for i in lists1[7][1:]]
#     pe = [ast.literal_eval(i) for i in lists1[2][1:]]
#     eopp = [ast.literal_eval(i) for i in lists1[4][1:]]
#     eodd = [ast.literal_eval(i) for i in lists1[6][1:]]

    
#     plt.plot(y, sp,  color = 'g', linestyle = '-', label = "SP")
#     plt.errorbar(y, sp, yerr=err1, fmt='go',capsize=5, elinewidth=2, markeredgewidth=2)
#     plt.plot(y, pe, color = 'b', linestyle = '--', label = "PE")
#     plt.errorbar(y, pe, yerr=err2,fmt='bo',capsize=10, elinewidth=2, markeredgewidth=2)
#     plt.plot(y, eopp, color = 'r', linestyle = ':',label = "EOpp")
#     plt.errorbar(y, eopp, yerr=err3, fmt='ro',capsize=15, elinewidth=3, markeredgewidth=2)
#     plt.plot(y, eodd,  color = 'y', linestyle = '-.', label = " EOdd")
#     plt.errorbar(y, eodd, yerr=err4, fmt='yo',capsize=20, elinewidth=2, markeredgewidth=2)

        
#     plt.xlabel('Num of requests')
#     plt.ylabel('unfairness')
#     plt.legend(loc ='upper right')
#     plt.title(title2, fontsize = 20)
#     plt.savefig('./results/summary/{}_{}_80%.png'.format( dataset,model))    
#     plt.show() 
    ################ accuracy plot###################################
    plt.figure(figsize=(10,4))
    x1 = lists[9][1:]
    x2 = lists[10][1:]
    
    x11 = lists1[9][1:]
    x22 = lists1[10][1:]
    translation = {39: None}

    acc1 = [ast.literal_eval(i) for i in x1]
    e1 = [ast.literal_eval(i) for i in x2]
    acc2 = [ast.literal_eval(i) for i in x11]
    e2 = [ast.literal_eval(i) for i in x22]
    
    plt.plot(y,acc1[1:],color = 'g', linestyle = '-', label = "acc without")
    plt.errorbar(y, acc1[1:], yerr=e1[1:],fmt='go',capsize=5, elinewidth=2, markeredgewidth=2)
    plt.plot(y, acc2,  color = 'b', linestyle = '--',    label = "acc with sisa")
    plt.errorbar(y, acc2, yerr=e2,fmt='bo',capsize=5, elinewidth=2, markeredgewidth=2)

    plt.xlabel('Num of requests')
    plt.ylabel('accuracy') 
    plt.legend(loc ='upper right')
    plt.title('accuracy result for {} trained using NN'.format(dataset,model), fontsize = 20)
    plt.savefig('./results/summary/accuracy_for_{}_{}.png'.format( dataset,model))    
    plt.show()