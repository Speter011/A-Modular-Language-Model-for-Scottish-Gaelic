train = 0.8
dev = 0.1

print(f"Train split:{train:.0%}")
print(f"Dev split:{dev:.0%}")
print(f"Test split:{1 - train - dev:.0%}")