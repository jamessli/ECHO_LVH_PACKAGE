# ECHO_LVH_PACKAGE
This repository is for devleopment and containment of the echo-lvh-project under Mayo Clinic.
**
QuickStart Guide: Echo LVH Pipeline****

**1 Contents**
	
	1. Contents
	2. Installation
	3. Directory Structure
	4. Training
	5. Testing
	6. Demo

**2 Installation**

This repo can be obtained by directly cloning the existing git repository. Create a directory in
which to store the downloaded package and run:

	# git clone https://github.com/jamessli/ECHO_LVH_PACKAGE/ my_directory

In the package directory, locate the requirements.txt file. This contains all of the necessary
dependencies utilized by the package. Install everything by running:

	# pip install -r requirements.txt

**3 Directory Structure**

The package will enable users to designate the directory(s) in which the input data and output data will be
read from and written to respectively. The input data must be in the form of video files contained within a
set directory hierarchy.

The directory structure must be formed as follows:

	my_directory/
		studies/
			test/
			      Amyloidosis/
			      HCM/
			      HTN/
	  train/
		Amyloidosis/
		HCM/
		HTN/
      
Within the individual disease labeled directories in the training directory, upload all of the training files in
the form of:

	study_name/
		video.mp4

These videos within the training data set will be automatically processed into a training/validation set split
at approximately 72:18 percentage. While initially, 10% of the total studies will be relocated to the test set
automatically.

In order to automate the setup process, run the following command with the target directory:

	#python main.py --prepare my_dir
	
Furthermore, if utilizing the pre-trained models is desired, download of these models from the online repository (Linked here: https://drive.google.com/file/d/1CtV3i2rPDO9wKVEEPcIlw0cy67o7KIbb/view?usp=sharing) can be done with the following:

	#python main.py --prepare my_dir --download
	
If the download flag is utilized during the prepare phase. The 'model' directory will be pre-populated with fully trained models.

**4 Training**

Training the models on the given training and test data sets can be done with a single command as
follows:

	#python main.py --train my_dir --device my_device

In this case, my_dir represents the directory containing the studies as outlined in the previous section
my_device is the name of any inference device/devices in the server in the form GPU:0, CPU:0…
Entering a single device as input will train on a single device. Entering multiple devices will attempt to
create a parallel training environment with those devices. Entering nothing will default to CPU:0. This
however, will likely fail with memory insufficiency unless a small batch size is provided.
In addition to training with the default commands as show, the package can also consume a
hyperparameters.txt file that contains hyperparameter values that overwrites the native ones provided in
the script. This file is of the format:

	batch_size = int
	epochs = int
	weight_decay = float
	learnable_layers = float
	learning_rate = float
	learning_rate_decay = int
	dropout_rate = float

Where:

	batch_size - the batch size
	epochs - the number of epochs of training
	weight decay - the severity of l2 regularization
	learnable_layers - the number of layers with tunable weights
	learning_rate - the initial learning rate of the optimizer
	learning_rate_decay – the number of epochs until the learning rate is halved
	dropout_rate – the severity of dropout chance after each layer.

**5 Testing**

Testing the models on the previously uploaded training and test data sets can be done with a single command as follows:

	#python main.py --test my_dir --device my_device
  
Where my_dir represents the location of the studies directory where the test and train directories are located.
To test this option with ablation, wherein the program will cycle between the view and slice-based ablations, run the command with an affirmative ablation flag as follows:

	#python main.py --test my_dir --device my_device –ablate yes
  
When running inferencing without having previously performed training, the program will automatically download the missing files from the network and will place them into the models directory.
	
Otherwise, in order to download all models from an online repo as mentioned in section 3.

	#python main.py --test my_dir --device my_device –ablate yes

**6 Demo**

A visual demonstration is available through streamlit that shows the inferencing and video generation in
real time on a single case study. This can be done through the following command.

	# streamlit run video_demo.py
  
At the top text box, enter the path to the case study and select the inference device from the drop-down menu.
