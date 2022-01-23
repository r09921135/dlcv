# HW2 Problem 1: GAN
In this problem, you should implement a GAN model from scratch and train it on the face dataset.

# HW2 Problem 2: ACGAN
In this problem, we aim to generate images for the corresponding digit inputs. You should implement an ACGAN model from scratch and train it on the mnistm dataset.

# HW2 Problem 3: DANN
In this problem, you need to implement DANN on digits datasets (USPS, MNIST-M and SVHN).

# HW2 Bonus: Improved UDA model
you need to modify your model in Problem 3 and redo the three scenarios in Problem 3-2. Note that your modified model should perform better on all three scenarios.

<p align="center">
  <img width="853" height="500" src="http://mmlab.ie.cuhk.edu.hk/projects/CelebA/intro.png">
</p>

For more details, please click [this link](https://drive.google.com/drive/folders/1loYdSncANJHv9qtIcb5Dsmp0ImcdIPn4?usp=sharing) to view the slides of HW2. **Note that all of hw2 videos and introduction pdf files can be accessed in your NTU COOL.**

# Usage
To start working on this assignment, you should clone this repository into your local machine by using the following command.

    git clone https://github.com/DLCV-Fall-2021/hw2-<username>.git
Note that you should replace `<username>` with your own GitHub username.

### Evaluation
To evaluate your UDA models in Problems 3 and Bonus, you can run the evaluation script provided in the starter code by using the following command.

    python3 hw2_eval.py $1 $2

 - `$1` is the path to your predicted results (e.g. `hw2_data/digits/mnistm/test_pred.csv`)
 - `$2` is the path to the ground truth (e.g. `hw2_data/digits/mnistm/test.csv`)

Note that for `hw2_eval.py` to work, your predicted `.csv` files should have the same format as the ground truth files we provided in the dataset as shown below.

| image_name | label |
|:----------:|:-----:|
| 00000.png  | 4     |
| 00001.png  | 3     |
| 00002.png  | 5     |
| ...        | ...   |

### Submission Format
Aside from your own Python scripts and model files, you should make sure that your submission includes *at least* the following files in the root directory of this repository:
 1.   `hw2_<StudentID>.pdf`  
The report of your homework assignment. Refer to the "*Grading*" section in the slides for what you should include in the report. Note that you should replace `<StudentID>` with your student ID, **NOT** your GitHub username.
 2.   `hw2_p1.sh`  
The shell script file for running your GAN model. This script takes as input a path and should output your 1000 generated images in the given path.
 3.   `hw2_p2.sh`  
The shell script file for running your ACGAN model. This script takes as input a path and should output your 1000 generated images in the given path.
 4.   `hw2_p3.sh`  
The shell script file for running your DANN model. This script takes as input a folder containing testing images and a string indicating the target domain, and should output the predicted results in a `.csv` file.
 5.   `hw2_bonus.sh`  
The shell script file for running your improved UDA model. This script takes as input a folder containing testing images and a string indicating the target domain, and should output the predicted results in a `.csv` file.

We will run your code in the following manner:

    bash ./hw2_p1.sh $1
    bash ./hw2_p2.sh $1
    bash ./hw2_p3.sh $2 $3 $4
    bash ./hw2_bonus.sh $2 $3 $4

-   `$1` is the path to your output generated images (e.g. `~/hw2/GAN/output_images` or `~/hw2/ACGAN/output_images`).
-   `$2` is the directory of testing images in the **target** domain (e.g. `~/hw2_data/digits/mnistm/test`).
-   `$3` is a string that indicates the name of the target domain, which will be either `mnistm`, `usps` or `svhn`. 
	- Note that you should run the model whose *target* domain corresponds with `$3`. For example, when `$3` is `mnistm`, you should make your prediction using your "SVHN→MNIST-M" model, **NOT** your "MNIST-M→SVHN" model.
-   `$4` is the path to your output prediction file (e.g. `~/test_pred.csv`).

> 🆕 ***NOTE***  
> For the sake of conformity, please use the `python3` command to call your `.py` files in all your shell scripts. Do not use `python` or other aliases, otherwise your commands may fail in our autograding scripts.

### Packages
This homework should be done using python 3.8. For a list of packages you are allowed to import in this assignment, please refer to the requirments.txt for more details.

You can run the following command to install all the packages listed in the requirements.txt:

    pip3 install -r requirements.txt

Note that using packages with different versions will very likely lead to compatibility issues, so make sure that you install the correct version if one is specified above. E-mail or ask the TAs first if you want to import other packages.

