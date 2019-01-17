# Python Video Pulserate using the chrominance method
Python implementation of a pulse rate monitor using rPPG from video with the chrominance method.
It uses OpenCV for face detection, for skin selection two methods are available. Skin classification using a hsv color range and forehead estimation. Due to the state of the art chrominance method the system is fairly motion robust. Furthermore, this framework also features a GUI that depicts the measured raw rPPG signal and the resulting fourier spectrum. 

![alt text](https://i.imgur.com/PsSnXq0.png) Video pulse rate captured while running on a threadmill.

# Dependencies
* Python3
* Numpy.
* Matplotlib. 
* OpenCV
* SciPy
For GUI:
* PyQT5
* PyQTgraph

# GUI

Execute the rPPG_GUI.py to launch the GUI. Edit the source variable in order to select the desired input. More user instructions can be found within the Python file.

# .Mat Extraction

The Raw rPPG signal from an offline recording can also be extracted to a .Mat file so it can be processed with Matlab. This is implemented in mat_exporter.py.

# Other files
rPPG_preprocessing.py contains some of the functions used for face tracking and skin classification.
rPPG_proccessing_realtime.py contains the signal processing (normalization, detrending, bandpass filtering and the chrominance method ) used to improve the signal. 
