import re

def test_sample(ground, response):
	evaluated = []
	total_recall = 0
	for i, r in enumerate(response):
		recall = 0
		g = ground[i][:-4]
		if g == re.sub(r"[\n\t\s]*", "", r):
			recall = 1
			total_recall += 1
		else:
			recall = 0
		evaluated.append({"tested": r, "ground": g,"recall": recall})
	return {"data": evaluated, "metrics": total_recall/len(response)}

# response = test_sample(['xdn65.png', 'pdw38.png', 'xce8d.png', '8g4yp.png', '74eyg.png', 'n8fp6.png', 'mcg43.png', 'ewnx8.png', '8gecm.png', 'nfcb5.png'], ['xdne65', 'pdw 38', 'exceed', '8g4yp', '74eyg', 'A8fpo6', 'mcg43', 'ewmx8', 'Budget', 'nfcD5']
# )
# print(response)