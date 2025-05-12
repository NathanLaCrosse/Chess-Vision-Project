import numpy as np
import cv2
import matplotlib.pyplot as plt
from pathlib import Path
from matplotlib.backend_bases import MouseButton
import sys
import io
import pandas as pd

# Mode sets which subset of the data we will be labelling
# Acceptable modes: 'Train', 'Test'
mode = 'Train'

path = "Data/" + mode + " Images/"
dat_folder = Path(path)

class LocationData:
    def __init__(self):
        self.locs = []
        self.clicks = 0

    def on_click(self, event):
        if event.inaxes and event.button is MouseButton.LEFT:
            self.locs.append(int(event.ydata))
            self.locs.append(int(event.xdata))

            self.clicks += 1
            if self.clicks >= 8:
                plt.close()
                self.clicks = 0

            print(f'click! at {(int(event.ydata), int(event.xdata))}')


loaded = pd.read_csv(f'Data/PointData{mode}.csv')
d = loaded.to_dict()

# Display the file using matplotlib to record points of interest
for file_dir in dat_folder.iterdir():
    # if loaded[file_dir] is not None:
    #     print("We already have this image!")
    #     continue

    s = str(file_dir)
    l = list(d['file_name'].values())


    if str(file_dir) in list(d['file_name'].values()):
        print("We already have this image!")
        continue

    im = cv2.imread(file_dir)[:,:,::-1]

    print(file_dir, "\n\n")
    plt.imshow(im)

    row_location_data = LocationData()
    plt.connect('button_press_event', row_location_data.on_click)
    plt.show()

    key = max(d['file_name'].keys()) + 1
    d['file_name'][key] = file_dir
    for i in range(8):
        d[f'point{i + 1}_row'][key] = row_location_data.locs[i*2]
        d[f'point{i + 1}_col'][key] = row_location_data.locs[i*2+1]

df = pd.DataFrame(d)
df.to_csv(f"Data/PointData{mode}.csv", index=False)