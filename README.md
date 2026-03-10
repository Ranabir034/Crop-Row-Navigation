# Visual Serving for Crop-Row Navigation 
# Internship Hiring Project for Salin247.com
# Completed By: Ranabir Saha
#		Ph.D. Student | Graduate Research Assistant
#		Department of Mechanical Engineering
#		Iowa State University
# For any inquiry, email: ranabir@iastate.edu

project_root/
│
├── crop_segmentation.py     # main detection script
├── crop_batch.sh            # batch-processing script
├── images/                  # input images
│     ├── ZED_imageimage_...png
│     └── ...
├── Results/                 # output folder (created automatically)
└── README.md


# Install the required Python Package
pip install numpy opencv-python


# To process one image and visualize the detected crop rows
python3 crop_segmentation.py \
    --image images/input.png \
    --out Results/single_out.png


Optional debug mode
python3 crop_segmentation.py \
    --image images/Reference.png \
    --out Results/debug_out.png \
    --debug


# Run for all images
./crop_batch.sh


# Thank you!
