import json
from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd

# Paths
metrics_file = Path("../results/coarse_full_evaluation_metrics.json")
output_file = Path("../results/coarse_tag_frequency.png")

# Load evaluation metrics
with metrics_file.open("r", encoding="utf-8") as f:
    metrics = json.load(f)

# Extract tag names and their frequencies
per_tag = metrics["per_tag"]

tags = list(per_tag.keys())
frequencies = [
    per_tag[tag]["support"] for tag in tags
]

# Sort from most frequent to least frequent
sorted_data = sorted(
    zip(tags, frequencies),
    key=lambda item: item[1],
    reverse=True
)

tags, frequencies = zip(*sorted_data)

# Create the graph
plt.figure(figsize=(16, 6))
plt.bar(tags, frequencies)

plt.xlabel("PoS tag")
plt.ylabel("Number of tokens")
plt.title("Frequency of Gold PoS Tags in the Test Set")

plt.xticks(rotation=90)
plt.tight_layout()

# Save and display
plt.savefig(output_file, dpi=300, bbox_inches="tight")
plt.show()


## 



## Show Most common errors
df = pd.read_csv("../results/coarse_full_predictions.csv")

aligned = df[df["alignment_status"] == "aligned"]
errors = aligned[aligned["gold_tag"] != aligned["predicted_tag"]]

confusions = (
    errors.groupby(["gold_tag", "predicted_tag"])
    .size()
    .reset_index(name="count")
    .sort_values("count", ascending=False)
)

# 20 most common mistakes
print(confusions.head(20).to_string(index=False))



