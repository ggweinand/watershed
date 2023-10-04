import matplotlib.pyplot as plt
import numpy as np
import sys

from PIL import Image

from watershed import watershed

if __name__ == "__main__":
    filename = sys.argv[1]
    image = np.array(Image.open(filename))
    result = watershed(image)
    plt.imshow(result, cmap='gray', vmin=0, vmax=255)
    plt.show()
