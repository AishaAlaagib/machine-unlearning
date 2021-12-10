import pandas as pd

import argparse

# Compute stats based on the execution time (cumulated feed-forward + backprop.) of the shards

parser = argparse.ArgumentParser()
parser.add_argument('--container', help="Name of the container")
parser.add_argument("--data", default="compas",  help='german_credit,adult_income, compas, default_credit, marketing')
parser.add_argument("--per", default=None, type=int,  help="Name of the req percentage")

args = parser.parse_args()

t = pd.read_csv('containerss/{}/{}/{}/times/times'.format(args.per,args.data, args.container), names=['time'])
print('{},{}'.format(t['time'].sum(),t['time'].mean()))