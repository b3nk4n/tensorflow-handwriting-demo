import numpy as np
import utils.ui

dialog = utils.ui.CanvasDialog("Read Handwriting...", 32, 32,
                               scale=5, num_letters=6)
data = dialog.show()
array = np.asarray(data)
print(array.shape)
