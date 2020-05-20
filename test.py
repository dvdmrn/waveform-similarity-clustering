import json
import numpy as np


with open("sparse_feature_matrix_graphemic.json","r") as f:
	featurematrix = json.load(f)

with open("pdata.json","r") as f:
	X = json.load(f)


print(X[0]["vocalizations"])
for g in X[0]["vocalizations"][0]:
	try:
		features=featurematrix[g]
		print(features)
	except:
		print(f"passing {g}")
