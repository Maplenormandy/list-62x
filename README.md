list-62x
========

Light Invariant Stable Features auto camera setting correction algorithm, for MIT 16.62x Fall 2014 / Spring 2015.

Please contact maplenormandy@gmail.com and aaron.phs@gmail.com for information on how to get the data associated with this project.

Note on installation. Make sure you're using OpenCV 2.4.9

    cd opencv
    mkdir release
    cd release
    cmake -D CMAKE_BUILD_TYPE=RELEASE -D CMAKE_INSTALL_PREFIX=/usr/local -D OPENCV_EXTRA_MODULES_PATH=../../opencv_contrib/modules ..
    
    make -j4
    sudo make install

Another helpful utility is Coriander, which is like cheese but for 1394 cameras

    sudo apt-get install coriander

Running
-------

To make it so you can run scripts as not root, do

    lsusb
    sudo chmod 666 /dev/bus/usb/002/001

Where 002 and 001 are replaced by the Bus and Device from lsusb. Then, you can run the script by doing

    python collectData.py fullrange.yaml
