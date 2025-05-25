import os
import json
import matplotlib.pyplot as plt

def list_images(folder, exts=(".jpg", ".png")):
    return [os.path.join(folder, f) for f in os.listdir(folder) if f.endswith(exts)]

def save_json(data, path):
    with open(path, 'w') as f:
        json.dump(data, f, indent=2)

def plot_distribution(counts, title="Distribution", save_path=None):
    plt.figure(figsize=(8, 4))
    plt.bar(counts.keys(), counts.values())
    plt.title(title)
    if save_path:
        plt.savefig(save_path)
    else:
        plt.show()