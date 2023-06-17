# AutoDrive GTA5
[source](https://github.com/DrakenWan/AutoDrive-GTA5)


 Autonomous Driving simulation in GTA 5 game

Temporarily made public. Will be privated soon.

### Requirements

- Your own instance of GTA 5 game
- Your python environment (anaconda preferably) and required libraries for a conda env are listed in `Requirements.txt` file. Link is [here](./Requirements/Requirements.txt)
- a GPU is preferable for training 
- You require installation of Native Trainer Mod in your GTA5. [Here](https://www.gta5-mods.com/tools/script-hook-v) is the link for it. The instructions are present on the website and also included in the download zip file.


### Training scenarios
- Currently, only considered training scenario where there is no traffic.
- Different routes were selected to generate data on these routes repeatedly.
- Below map shows highlighted routes on which data was generated:-
 ![Map](./Snapshots/Map_route_scenarios.png)

- Due to time constraints, we were able to generate just 1.5 GB worth of data. The model requires a huge amount of data to have enough values for `rare` inputs such as `reverse()` or take into consideration `rare cases` such as turns or sharp turns.

### Workflow / Steps to recreate
1.  First, ensure that you are playing the gta game in windowed 800x600 resolution. Move the game window to the top left.
2. Now open your conda shell, activate your environment with all requirements installed.
3. **Collecting data**: for data collection you need to run, `python collect_data.py 0` or `python collect_data.py 1`. At option `1`, the data collection introduces noise/error in real time to the output labels instead of the `no input` label. If `0` is selected then `no input` option is taken into consideration. You can use the above map for reference to generate data in the same scenarios. We also links to our own data collection on google drive. [Here is the link to the folder with the data](https://drive.google.com/drive/folders/1azfugKU4pLni8HcxljjdaoFlpzhjOmu-?usp=sharing). You can find data in `Data` folder as `.npy` files.
    - Note: if you want to pause for a break during data collection, you can toggle the `T` key to pause/unpause.
    - To **visualize the collected data**, you can run `python visualise_training_data.py`. This will open a cv2 window showing entire sequence of the collected training data along with key labels in cmd. If you want to quit you can press `q` button while cv2 window is active.

4. **Preprocess data**: This part is necessary to be run before the subsequent steps. It changes the data format in the saved numpy arrays for efficient processing during training and other steps that come after.

5. **Model Training**: For training just run `python train_model.py`. This will take on average a time of 2 hrs on 1.5 GB worth of data. If you want you can explicitly increase the `batch_size` hyperparameter to increase the `time/epoch`. Epochs can be increased too but it will take a huge amount of time (in scale of days).

6.  **Deployment**: To test our model in deployment. Keep the game window on top left. Run command `python model_test.py`. A countdown will start from 4 for you to recuperate your attention back to the game screen. You can toggle `T` key to pause/unpause the simulation of the inputs in the game.


If you do not have the ability to play GTA5, you can atleast work with step 3b to visualise the data. You can run Step 4 as well to see class distribution stats. You can train the model in Step 5. But Step 6 does require that you have an instance of gta 5 running.


### Saved Data and Model

- Model weights are saved [here](./ModelSaves/inceptv3_model.pth)
- Training data can be found [here](https://drive.google.com/drive/folders/1IJPymguhy9dnGSijwE7e7idwzmnaMqNJ?usp=drive_link)




#### Working with NativeTrainer

- Pressing `F4` to toggle the NativeTrainer mod menu options in the game
- Go to `Vehicle spawne`r in `Vehicles` option. Go to list 4/6 and spawn "Pegassi" bike.
- You can use `teleport to marker` option in `Locations` tab to spawn to different routes quickly
- No traffic scenario was simulated for data collection. Traffic lights were considered during data collection.





### Challenges

- Capturing the screen with high FPS value. Initially, we used ImageGrab from PIL library to perform the screen capture and then visualise and perform feature engineering using OpenCV library. However, the FPS was at a dwindling rate of `10-15`. Afterwards, we made use of `grabscreen` API which we build using win32gui and win32api.<sup><abbr title="Details and references for code are in the grab_screen.py file.">Note</abbr></sup>. This led to a dramatic increase in FPS to `83.5 fps` to `100 fps`.

- Feature Engineering was one part that was initially done manually. We generated an `ROI` mask, added edge detection and lane generation. These additions were redundant for the deep learning model as it has the ability to do feature engineering itself. Also this process was hindering with the FPS capture in the game. It was necessary things such as image resizing and ROI calculation be performed along with the capture. This part added extra redundancy and was later removed

- Data Volume. The size of data collected is small compared to wide variety of scenarios and routes that might occur in the game. We reduced our data collection to specific scenarios and routes only. The model requires large amount of data to perform well in simulation.

- With regards to class imbalance in our data, we performed reduction of values in classe with large number values down to the numbers for class with lowest values. This led to decrease in data and bias in model. We avoided performing class balance and it led to one output dominating the simulation. So we tried to perform class balancing in the data in real time and used `python balance_data.py` simultaneously to check the class balance statistics and adjust for it while data collection.

- Selecting Ground Truth Labels. We used limited ground truth labels to consider the movements - `Forward Left`, `Forward`, `Forward Right`, `Stop/Reverse` and `No Input`. We tried our best to capture these movements using the combinations of the keys W, A, S and D. Other input events or ECU data/events were not considered because of lack of better ways to capture such information without hindering the image data capture process.

### Comparative Analysis

- This is more qualitative.
- I will be comparing it with SentDex's implementation (update upto video #14).
- Our model is Google's Inception v3 whereas the related work makes use of AlexNet
- There are some qualitative points to be considered when comparing both models:-
    - Inception model is much more complex and is deeper. This complexity helps it capture more complex representations
    - Inception model was performing better than AlexNet and it has the capability to generalise better on the data
- However, the quality of training also depends heavily on fine tuning process for the game as well as the quality of the data that we collect.


### Areas of Improvement
- In continuation to last section
- Based on our evaluation, it seems the more data we have the more better it is. However, the larger data will require taking into consideration modifying the preprocessing part to accomodate for large scale image data.
- Increasing training set size will require more training time. For instance, training nearly 2GB of collected training data on just few route scenarios takes the model 4-5 minutes at minimum per epoch. The model is being trained for 25 epochs in average. So the total (minm) training time is at an estimate of 2 hours. 
- We are performing Minibatch optimization, one way to speedup process is to increase batch size for training part. However, need to take into consideration the GPU memory constraints.




### Files

Briefly explains all the files that are being used in the project below.

- **collect_data.py**: This file helps collect data from the game. The file takes one input argument from choices in {0,1}. If 1 option is selected then noise is introduced into the labels insteadd of `no input` label option. Data is saved in disk after every 500 captures. Data collection can be paused by toggling `T` key.

- **visualise_training_data.py**: This file is used to run a sequence of training data captured so far.

- **balance_data.py**: Mainly a redundant file as of now. Can be used to check the class label distribution statistics.

- **inception_v3.py**: Contains the Inception v3 model definition in pytorch

- **train_data.py**: trains our inception 3 model ising minibatch optimization and evaluates it on validation data. Saves the model weights at the end of training.

- **directkeys.py**: Makes use of ctypes library to account for keyboard events such as KeyPress or KeyRelease

- **getkeys.py**: Captures the key that was pressed and returns a list of those values

- **grabscreen.py**: An API built over win32gui and win32api to capture the game screen

- **model_test.py**: Deploy our trained model in the game and simulate the AI.
