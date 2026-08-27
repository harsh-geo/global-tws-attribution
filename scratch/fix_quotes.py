import re
with open('c:/SILIKA/Thesis/data2/src/modeling/step04_run_attribution.m', 'r') as f:
    text = f.read()
text = text.replace("''", "'")
with open('c:/SILIKA/Thesis/data2/src/modeling/step04_run_attribution.m', 'w') as f:
    f.write(text)
