

from dataloading.SpeechDatasetLoader import SpeechDatasetLoader
from dataloading.SpeechDataset import SpeechDataset


tedlium = SpeechDataset("LIUM/tedlium", "release1")

dataset_objs = [tedlium]

loader = SpeechDatasetLoader(dataset_objs)

test_loader, val_loader, test_loader = loader.load_datasets(batch_size=256)

# print size of dataset
print(len(test_loader.dataset))
print(len(val_loader.dataset))
print(len(test_loader.dataset))

