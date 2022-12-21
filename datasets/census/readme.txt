Census income classification

http://archive.ics.uci.edu/ml/datasets/Census+Income


Label:
Negative : person earns lees than or equal to 50K
Positive : person earns more than 50k

Rules:

For this case we created the rules synthetically as follows:
We hold out disjoint 16k random points from the training dataset as a proxy for human knowledge and extract a PART decision list from it as our set of rules. We retain only
those rules which fire on L.

https://mathematicaforprediction.wordpress.com/2014/03/30/classification-and-association-rules-for-census-income-data/




