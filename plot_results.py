import json
import matplotlib.pyplot as plt
import os
import numpy as np

RESULTS_DIR = r'C:\Users\Y\Desktop\Image-Classification-System\results'
HISTORY_FILE = os.path.join(RESULTS_DIR, 'history.json')

with open(HISTORY_FILE, 'r') as f:
    data = json.load(f)

models = list(data.keys())
metrics = ['accuracy', 'val_accuracy', 'loss', 'val_loss']
titles = ['Training Accuracy', 'Validation Accuracy', 'Training Loss', 'Validation Loss']
ylabels = ['Accuracy', 'Accuracy', 'Loss', 'Loss']
filenames = ['comparison_accuracy.png', 'comparison_val_accuracy.png', 'comparison_loss.png', 'comparison_val_loss.png']

for metric, title, ylabel, filename in zip(metrics, titles, ylabels, filenames):
    plt.figure(figsize=(10, 6))
    for model in models:
        values = data[model].get(metric, [])
        epochs = range(1, len(values) + 1)
        plt.plot(epochs, values, label=model)
    
    plt.title(title)
    plt.xlabel('Epochs')
    plt.ylabel(ylabel)
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(RESULTS_DIR, filename))
    plt.close()

test_accs = [data[model].get('test_accuracy', 0) for model in models]
plt.figure(figsize=(8, 6))
plt.bar(models, test_accs, color=['blue', 'orange', 'green'])
plt.title('Test Accuracy Comparison')
plt.ylabel('Accuracy')
plt.ylim(0, 1)
for i, v in enumerate(test_accs):
    plt.text(i, v + 0.01, f'{v:.2%}', ha='center')
plt.savefig(os.path.join(RESULTS_DIR, 'comparison_test_accuracy.png'))
plt.close()
