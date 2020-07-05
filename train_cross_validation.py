import tensorflow as tf
import nibabel as nib
import os
from miscnn.processing.subfunctions.normalization import Normalization
from miscnn.processing.subfunctions.clipping import Clipping
from miscnn.processing.subfunctions.resampling import Resampling
from miscnn.data_loading.interfaces.nifti_io import NIFTI_interface
from miscnn.data_loading.data_io import Data_IO
from miscnn.neural_network.model import Neural_Network
from miscnn.neural_network.metrics import dice_soft, dice_crossentropy, tversky_loss
from miscnn.evaluation.cross_validation import cross_validation
from miscnn.processing.data_augmentation import Data_Augmentation
from miscnn.processing.preprocessor import Preprocessor
from tensorflow.keras.callbacks import ReduceLROnPlateau

# Tensorflow Configurations
gpu_options = tf.compat.v1.GPUOptions(allow_growth=True)
sess = tf.compat.v1.Session(config=tf.compat.v1.ConfigProto(gpu_options=gpu_options))
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3,4,5"

interface = NIFTI_interface(pattern="case_00[0-9]*", channels=1, classes=3)
# the path of the kits19 dataset
data_path = "../kits19/data/"

# check samples
for sample in os.listdir(data_path):
    if sample in ["LICENSE", "kits.json"]:
        continue

    print("Checking sample:", sample)
    path_vol = os.path.join(data_path, sample, "imaging.nii.gz")
    vol = nib.load(path_vol)
    vol_data = vol.get_data()

# Create the Data I/O object
data_io = Data_IO(interface, data_path)

sample_list = data_io.get_indiceslist()
sample_list.sort()
print("All samples: " + str(sample_list))

# Create and configure the Data Augmentation class
data_aug = Data_Augmentation(cycles=2, scaling=True, rotations=True, elastic_deform=True, mirror=True,
                             brightness=True, contrast=True, gamma=True, gaussian_noise=True)

# Select Subfunctions for the Preprocessing

# Create a pixel value normalization Subfunction through Z-Score
sf_normalize = Normalization()
# Create a clipping Subfunction between -79 and 304
sf_clipping = Clipping(min=-79, max=304)
# Create a resampling Subfunction to voxel spacing 3.22 x 1.62 x 1.62
sf_resample = Resampling((3.22, 1.62, 1.62))

# Assemble Subfunction classes into a list Be aware that the Subfunctions will be exectued according to the list order!
subfunctions = [sf_resample, sf_clipping, sf_normalize]

# Create and configure the Preprocessor class
pp = Preprocessor(data_io, data_aug=data_aug, batch_size=2, subfunctions=subfunctions, prepare_subfunctions=True,
                  prepare_batches=False, analysis="patchwise-crop", patch_shape=(80, 160, 160))
# Adjust the patch overlap for predictions
pp.patchwise_overlap = (40, 80, 80)

# Create the Neural Network model
model = Neural_Network(preprocessor=pp, loss=tversky_loss, metrics=[dice_soft, dice_crossentropy],
                       batch_queue_size=3, workers=3, learninig_rate=0.0001)

# Define Callbacks
cb_lr = ReduceLROnPlateau(monitor='loss', factor=0.1, patience=20, verbose=1, mode='min', min_delta=0.0001, cooldown=1,
                          min_lr=0.00001)

# Exclude suspious samples from data set
del sample_list[133]
del sample_list[125]
del sample_list[68]
del sample_list[37]
del sample_list[23]
del sample_list[15]

# Create the validation sample ID list
validation_samples = sample_list[0:120]
print("Validation samples: " + str(validation_samples))

# Perform a 3-fold Cross-Validation
cross_validation(
    validation_samples,
    model,
    k_fold=3,
    epochs=70,
    iterations=150,
    evaluation_path="evaluation",
    draw_figures=True,
    callbacks=[cb_lr],
    run_detailed_evaluation=True
)
