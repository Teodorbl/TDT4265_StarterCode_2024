# Working on Cybele Computers

There are 25 computers in Cybele (the lab previously known as Tulipan), all with powerful GPU cards (RTX 4090). There are a lot of students in the class and you have to consider your fellow students when you are using the GPU resources.

We ask you to follow this rule when using the computers:

- Each group can only use a single computer at a time.

# Working remotly via SSH (18:00 - 08:00 only!)

Note that you need to be connected to the NTNU VPN if you are outside the NTNU local network for the connection to be successful. Se this link for information about the VPN: https://i.ntnu.no/wiki/-/wiki/English/Install+vpn

You are also allowed to use the Cybele computers remotely using SSH. 

To log into the nodes via ssh you can use the following command in the terminal:

```bash
ssh <USERNAME>@clab[01-25].idi.ntnu.no
```
Change out [01-25] with one of the 25 nodes, and your username.
Another option is to connect directly via VS Code through the "Remote - SSH" extension and log in via the same command.


The computers can not however be used during school time: 08:00 - 18:00 and you should check that no one is using the GPU before you start training.
This can be done by checking the utilization of the GPU with the following command:

```bash
nvidia-smi
```
There might be some processes using the GPU for the GUI in linux but make sure the utilization is less than 10% before using it!
Here is an example of how the output is when the GPU is not in use:
```bash
+---------------------------------------------------------------------------------------+
| NVIDIA-SMI 535.161.07             Driver Version: 535.161.07   CUDA Version: 12.2     |
|-----------------------------------------+----------------------+----------------------+
| GPU  Name                 Persistence-M | Bus-Id        Disp.A | Volatile Uncorr. ECC |
| Fan  Temp   Perf          Pwr:Usage/Cap |         Memory-Usage | GPU-Util  Compute M. |
|                                         |                      |               MIG M. |
|=========================================+======================+======================|
|   0  NVIDIA GeForce RTX 4090        On  | 00000000:01:00.0 Off |                  Off |
|  0%   31C    P8               9W / 450W |    111MiB / 24564MiB |      0%      Default |
|                                         |                      |                  N/A |
+-----------------------------------------+----------------------+----------------------+

+---------------------------------------------------------------------------------------+
| Processes:                                                                            |
|  GPU   GI   CI        PID   Type   Process name                            GPU Memory |
|        ID   ID                                                             Usage      |
|=======================================================================================|
|    0   N/A  N/A      1650      G   /usr/lib/xorg/Xorg                           85MiB |
|    0   N/A  N/A      2140      G   /usr/bin/gnome-shell                         16MiB |
+---------------------------------------------------------------------------------------+
```
## Environment
Every computer in the cybele lab comes with python2 and python3.
You can run code in python3 by using, `python3 my_program.py`.


### Installing packages
All the packages required to run the assignment is allready installed on the computers.
If you want to install additional packages in your environment, install it locally. For example:

```bash
pip3 install --user package_name
```

To get a list of already installed packages, use:
```
pip list
```

Pytorch is already installed on the computers and it should work out of the box with the GPU. Just launch python with "python3" in the terminal.
