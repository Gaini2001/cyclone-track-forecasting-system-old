from src.training.train import load_feature_dataset
from src.preprocessing.create_sequences import create_sequences

df = load_feature_dataset()

X, y, sids = create_sequences(df)

print("X shape:", X.shape)
print("y shape:", y.shape)
print("SID shape:", sids.shape)

print("\nUnique storms:")
print(len(set(sids)))