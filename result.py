import miscnn
from miscnn.data_loading.interfaces.nifti_io import NIFTI_interface
from miscnn.utils.visualizer import visualize_evaluation

interface = NIFTI_interface(pattern="case_000[0-9]*", channels=1,
                            classes=3)

# Create the Data I/O object
data_path = "../kits19/data/"
data_io = miscnn.Data_IO(interface, data_path)
sample_list = data_io.get_indiceslist()
sample_list.sort()
print("All samples: " + str(sample_list))

# Load the sample
sample = data_io.sample_loader(sample_list[24], load_seg=True, load_pred=True)
# Access image, truth and predicted segmentation data
img, seg, pred = sample.img_data, sample.seg_data, sample.pred_data
# Visualize the truth and prediction segmentation as a GIF

visualize_evaluation(sample_list[24], img, seg, pred, "plot_directory/")
