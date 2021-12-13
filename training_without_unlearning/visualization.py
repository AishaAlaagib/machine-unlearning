
import matplotlib.pyplot as plt
import csv
from csv import reader
import ast


lists = [[] for _ in range(6)]
lists1 = [[] for _ in range(6)]

summary = [[] for _ in range(6)]
datasets = ('adult_income','marketing')#,'new_adult_income')#, )# 'default_credit',)
# pers = (50)
y = ['1%','5%','10%','15%', '20%', '25%', '30%', '35%', '40%','45%', '50%','55%', '60%','65%','70%','75%','80%','85%','90%','95%']
# dataset = 'marketing'
model = 'DNN'
for dataset in datasets:
    per  = 50
    file = './results/summary/summary_{}_{}.csv'.format(dataset,model)
    file2 = '../results/{}/{}.csv'.format(per, dataset)

    title = 'unfairness_summary_for_{}_{}'.format(dataset,model)
    csv.register_dialect(
    'mydialect',
    delimiter = ',',
    quotechar = '"',
    doublequote = True,
    skipinitialspace = True,
    quoting = csv.QUOTE_MINIMAL)
    with open(file2, 'r') as read_obj:
    #         lines = reader(read_obj)
        lines = csv.reader(read_obj, dialect='mydialect')

        for row in lines:
            print(row)
            lists1[0].append(row[3])
            lists1[1].append(row[4])
            lists1[2].append(row[5])
            lists1[3].append(row[6])
            lists1[4].append(row[1]) #y
            lists1[5].append(row[2]) # acc

        print(lists)
    with open(file, 'r') as read_obj:
    #         lines = reader(read_obj)
        lines = csv.reader(read_obj, dialect='mydialect')

        for row in lines:
            print(row)
            lists[0].append(row[5])
            lists[1].append(row[6])
            lists[2].append(row[7])
            lists[3].append(row[8])
            lists[4].append(row[2]) #y
            lists[5].append(row[4]) # acc

        print(lists)
    #     summary[0].append(lists[0][-1])
    #     summary[1].append(lists[1][-1])
    #     summary[2].append(lists[2][-1])
    #     summary[3].append(lists[3][-1])
    #     summary[4].append(lists[4][-1])
    #     summary[5].append(lists[5][-1])

    # Increase the width
    s = [ast.literal_eval(i) for i in lists[0][1:]]
    d = [ast.literal_eval(i) for i in lists[1][1:]]
    f = [ast.literal_eval(i) for i in lists[2][1:]]
    g = [ast.literal_eval(i) for i in lists[3][1:]]
    # y = [ast.literal_eval(i) for i in lists[4][1:]]

#     print(s)
#     print(y)
    plt.figure(figsize=(15,4))
    plt.plot(y, s,  color = 'g', linestyle = '-',
             label = "SP")
    plt.plot(y, d, color = 'r', linestyle = '--',
             label = "PE")
    plt.plot(y, f, color = 'b', linestyle = ':',
             label = "EOpp")
    plt.plot(y, g,  color = 'y', linestyle = '-.',
             label = " EOdd")
    #     plt.xticks(rotation = 25)
    plt.xlabel('Num of requests')
    plt.ylabel('unfairness')
    plt.legend()
    plt.title(title, fontsize = 20)
    plt.savefig('./results/{}_{}.png'.format( dataset,model))    
    plt.show() 

    ################ accuracy plot###################################
    plt.figure(figsize=(10,4))
    x1 = lists[5][1:]
    x2 = lists1[5][1:]
    translation = {39: None}
    # Printing list using translate Method
    print ("Printing list without quotes", str(x1).translate(translation))
    print ("Printing list without quotes", str(x2).translate(translation))

    x1 = [ast.literal_eval(i) for i in x1]
    x2 = [ast.literal_eval(i) for i in x2]

    print(x1,x2)
    # x = sorted(x)
    # print(x)
    plt.plot(y, x1,  color = 'b', linestyle = '-',    label = "acc after unlearning")
    plt.plot(y, x2,  color = 'g', linestyle = '-',    label = "acc without unlearning")

    plt.xlabel('Num of requests')
    plt.ylabel('accuracy')
    plt.legend()
    plt.title('accuracy_summary_for_{}_{}'.format(dataset,model), fontsize = 20)
    plt.savefig('./results/accuracy_for_{}_{}.png'.format( dataset,model))    
    plt.show()