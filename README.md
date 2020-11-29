# YOLOv4 object detection

These guidelines are about how to install deep learning dependencies and train Yolo modules on Ubuntu 18.04.

## Dependencies

- Ubuntu
- NVIDIA GPUs
- NVIDIA graphic driver
- CUDA toolkit
- cuDNN library

### 1. Install Ubuntu dependencies

```
$ sudo apt-get update
$ sudo apt-get upgrade
```

Let’s install development tools, image and video I/O libraries, GUI packages, optimization libraries, and other packages:

```
$ sudo apt-get install build-essential cmake unzip pkg-config
$ sudo apt-get install libxmu-dev libxi-dev libglu1-mesa libglu1-mesa-dev
$ sudo apt-get install libjpeg-dev libpng-dev libtiff-dev
$ sudo apt-get install libavcodec-dev libavformat-dev libswscale-dev libv4l-dev
$ sudo apt-get install libxvidcore-dev libx264-dev
$ sudo apt-get install libgtk-3-dev
$ sudo apt-get install libopenblas-dev libatlas-base-dev liblapack-dev gfortran
$ sudo apt-get install libhdf5-serial-dev
$ sudo apt-get install python3-dev python3-tk python-imaging-tk
```

CUDA 10 requires gcc v7 but Ubuntu 18.04 ships with gcc v7. If gcc v7 is not installed then need to install gcc and g++ v7:

```
$ sudo apt-get install gcc-7 g++-7
```

### 2. Install latest NVIDIA drivers (GPU only).

Let’s go ahead and add the NVIDIA PPA repository to Ubuntu’s Aptitude package manager:

```
$ sudo add-apt-repository ppa:graphics-drivers/ppa
$ sudo apt-get update
```

Now we can very conveniently install our NVIDIA drivers

```
$ sudo apt install nvidia-driver-xxx
```

Go ahead and reboot so that the drivers will be activated as your machine starts:

```
$ sudo reboot now
```

You’ll want to verify that NVIDIA drivers have been successfully installed:

```
$ nvidia-smi
```

### 3. Install CUDA Toolkit and cuDNN (GPU only)

Head to the NVIDIA developer website for CUDA 10.0 downloads.  
You can access the downloads via this direct link: https://developer.nvidia.com/cuda-10.0-download-archive

Make the following selections from the CUDA Toolkit download page:

- “Linux”
- “x86_64”
- “Ubuntu”
- “18.04”
- “runfile (local)”
- “Base Installer”

Click right button of your mouse on "Download" and choose "Copy link"
and use the wget command to download the runfile (be sure to copy the full URL):

```
$ wget https://developer.nvidia.com/compute/cuda/10.0/Prod/local_installers/cuda_10.0.130_410.48_linux
```

From there go ahead and install the CUDA Toolkit.

```
$ chmod +x cuda_10.0.130_410.48_linux.run
$ sudo ./cuda_10.0.130_410.48_linux.run --override
```

During installation, you will have to:

- Use “space” to scroll down and accept terms and conditions.
- Select y for “Install on an unsupported configuration”.
- Select n for “Install NVIDIA Accelerated Graphics Driver for Linux-x86_64 384.81?”
- Keep all other default values (some are y and some are n ). For paths, just press “enter”.

Now we need to update our ~/.bashrc file to include the CUDA Toolkit:

```
$ nano ~/.bashrc
```

Scroll to the bottom and add following lines:

```
# NVIDIA CUDA Toolkit
export PATH=/usr/local/cuda-10.0/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda-10.0/lib64
```

To save and exit with nano, simply press “ctrl + o”, then “enter”, and finally “ctrl + x”.
Once you’ve saved and closed your bash profile, go ahead and reload the file:

```
$ source ~/.bashrc
```

From there you can confirm that the CUDA Toolkit has been successfully installed:

```
$ nvcc -V
```

### 4. Install cuDNN (CUDA Deep Learning Neural Network library) (GPU only)

For this step, you will need to create an account on the NVIDIA website + download cuDNN.

Here’s the link: https://developer.nvidia.com/cudnn

When you’re logged in and on that page, go ahead and make the following selections:

- “Download cuDNN”
- Login and check “I agree to the terms of service fo the cuDNN Software License Agreement”
- “Archived cuDNN Releases”
- “cuDNN v7.4.1 for CUDA 10.0”
- “cuDNN Library for Linux”

The following commands will install cuDNN in the proper locations on your Ubuntu 18.04 system:

```
$ cd ~
$ tar -zxf cudnn-10.0-linux-x64-v7.4.1.x.tgz
$ cd cuda
$ sudo cp -P lib64/* /usr/local/cuda/lib64/
$ sudo cp -P include/* /usr/local/cuda/include/
$ cd ~
```

Above, we have:

- Extracted the cuDNN 10.0 v7.4.1.x file in our home directory.
- Navigated into the cuda/ directory.
- Copied the lib64/ directory and all of its contents to the path shown.
- Copied the include/ folder as well to the path shown.
- Take care with these commands as they can be a pain point later if cuDNN isn’t where it needs to be.

### 5. Configure Python virtual environment

First, let’s install pip, a Python package management tool:

```
$ wget https://bootstrap.pypa.io/get-pip.py
$ sudo python3 get-pip.py
```

Now that pip is installed, let’s go ahead and install the two virtual environment tools that I recommend — virtualenv and virtualenvwrapper:

```
$ python3 -m pip install virtualenv virtualenvwrapper
```

We’ll need to update our bash profile with some virtualenvwrapper settings to make the tools work together.

Go ahead and open your ~/.bashrc file using your preferred text editor again and add the following lines at the very bottom:

```
# virtualenv and virtualenvwrapper
export WORKON_HOME=$HOME/.virtualenvs
export VIRTUALENVWRAPPER_PYTHON=/usr/bin/python3
source /usr/local/bin/virtualenvwrapper.sh
```

And let’s go ahead and reload our ~/.bashrc file:

```
$ source ~/.bashrc
```

The virtualenvwrapper tool now has support for the following terminal commands:

- mkvirtualenv : Creates a virtual environment.
- rmvirtualenv : Removes a virtual environment.
- workon : Activates a specified virtual environment. If an environment isn’t specified all environments will be listed.
- deactivate : Takes you to your system environment. You can activate any of your virtual environments again at any time

Creating the my_environment_name environment

```
$mkvirtualenv my_environment_name -p python3
```

If your environment is not active, simply use the workon command:

```
$ workon my_environment_name
```

### 6. Install Python libraries

Now that our Python virtual environment is created and is currently active, let’s install NumPy and OpenCV using pip:

```
$ pip install numpy
$ pip install opencv-contrib-python
```

Let’s install libraries required for additional computer vision, image processing, and machine learning as well:

```
$ pip install scipy matplotlib pillow
$ pip install imutils h5py requests progressbar2
$ pip install scikit-learn scikit-image
```

### 7. Install TensorFlow for Deep Learning for Computer Vision with Python

```
$pip install tensorflow-2.0.0
```

Go ahead and verify that TensorFlow is installed in your my_environment_name virtual environment:

```
$ python
>>> import tensorflow
>>>
```

## Darknet (YOLO)
