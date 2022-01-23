
# HW1 Problem 1― Image Classification
In HW1 problem 1, you will need to implement an image classification model and answer some questions in the report.

# HW1 Problem 2― Semantic Segmentation
In HW1 problem 2, you will need to implement two semantic segmentation models and answer some questions in the report.

For more details, please click [this link](https://docs.google.com/presentation/d/1H4O5NrEK-AzS2jRggWSpzs9hHEsYlCNYQdJbHlDfH48/edit?usp=sharing) to view the slides of HW1

# Usage
To start working on this assignment, you should clone this repository into your local machine by using the following command.

    git clone https://github.com/DLCV-Fall-2021/hw1-<username>.git
Note that you should replace `<username>` with your own GitHub username.

### Evaluation
To evaluate your semantic segmentation model, you can run the provided evaluation script provided in the starter code by using the following command.

    python3 mean_iou_evaluate.py <--pred PredictionDir> <--labels GroundTruthDir>

 - `<PredictionDir>` should be the directory to your predicted semantic segmentation map (e.g. `hw1_data/prediction/`)
 - `<GroundTruthDir>` should be the directory of ground truth (e.g. `hw1_data/validation/`)

Note that your predicted segmentation semantic map file should have the same filename as that of its corresponding ground truth label file (both of extension ``.png``).

### Visualization
To visualization the ground truth or predicted semantic segmentation map in an image, you can run the provided visualization script provided in the starter code by using the following command.

    python3 viz_mask.py <--img_path xxxx_sat.jpg> <--seg_path xxxx_mask.png>

### Submission Format
Aside from your own Python scripts and model files, you should make sure that your submission includes *at least* the following files in the root directory of this repository:
 1.   `hw1_<StudentID>.pdf`  
The report of your homework assignment. Refer to the "*Submission*" section in the slides for what you should include in the report. Note that you should replace `<StudentID>` with your student ID, **NOT** your GitHub username.
 2.   `hw1_1.sh`  
The shell script file for running your classification model.
 3.   `hw1_2.sh`  
The shell script file for running your semantic segmentation model.

We will run your code in the following manner:

    bash hw1_1.sh $1 $2
where `$1` is the testing images directory (e.g. `test/images/`), and `$2` is the path of folder where you want to output your prediction file (e.g. `test/label_pred/` ). Please do not create the output prediction directory in your bash script or python codes.

    bash hw1_2.sh $1 $2
where `$1` is the testing images directory (e.g. `test/images/`), and `$2` is the output prediction directory for segmentation maps (e.g. `test/label_pred/` ). Please do not create the output prediction directory in your bash script or python codes.

### Packages
This homework should be done using python3.6. For a list of packages you are allowed to import in this assignment, please refer to the requirments.txt for more details.

You can run the following command to install all the packages listed in the requirements.txt:

    pip3 install -r requirements.txt

Note that using packages with different versions will very likely lead to compatibility issues, so make sure that you install the correct version if one is specified above. E-mail or ask the TAs first if you want to import other packages.
