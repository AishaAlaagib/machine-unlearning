import pandas as pd

import argparse

# Compute stats based on the execution time (cumulated feed-forward + backprop.) of the shards

parser = argparse.ArgumentParser()
parser.add_argument('--container', help="Name of the container")
parser.add_argument("--data", default="compas",  help='german_credit,adult_income, compas, default_credit, marketing')
parser.add_argument("--per", default=None, type=int,  help="Name of the req percentage")
parser.add_argument("--rseed", default=0, type=int,  help="random seed")

args = parser.parse_args()

t = pd.read_csv('containerss/{}/{}/{}/{}/times/times'.format(args.per,args.rseed,args.data, args.container), names=['time'])
print('{},{}'.format(t['time'].sum(),t['time'].mean()))