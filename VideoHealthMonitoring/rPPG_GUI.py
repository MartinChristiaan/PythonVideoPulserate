# Author : Martin van Leeuwen
# Video pulse rate monitor using rPPG with the chrominance method.

############################## User Settings #############################################

class Settings():
    def __init__(self):
        self.use_classifier = True  # Toggles skin classifier
        self.use_flow = False       # (Mixed_motion only) Toggles PPG detection 
                                    # with Lukas Kanade optical flow  
        self.show_cropped = True    # Shows the processed frame on the aplication instead of the regular one.
        self.sub_roi = []#          [.35,.65,.05,.15] # If instead of skin classifier, forhead estimation should be used
                                    # set to [.35,.65,.05,.15]
        self.use_resampling = True  # Set to true with webcam 
        
# In the source either put the data path to the image sequence/video or "webcam"
#source = "C:\\Users\\marti\\Downloads\\Data\\13kmh.mp4" # stationary\\bmp\\"
#source = "C:\\Users\\marti\\Downloads\\Data\\stationary\\bmp\\"
source = "webcam"
fs = 20 # Please change to the capture rate of the footage.

############################## APP #######################################################

from PyQt5 import QtGui  
from PyQt5 import QtCore  
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import *
from util.qt_util import *
from util.pyqtgraph_util import *
import numpy as np
from util.func_util import *
import matplotlib.cm as cm
from util.style import style
from util.opencv_util import *
from rPPG_Extracter import *
from rPPG_lukas_Extracter import *
from rPPG_processing_realtime import extract_pulse

## Creates The App 

fftlength = 300

f = np.linspace(0,fs/2,fftlength/2 + 1) * 60;
settings = Settings()


def create_video_player():
    frame = cv2.imread("placeholder.png")
    vb = pg.GraphicsView()
    frame,_ = pg.makeARGB(frame,None,None,None,False)
    img = pg.ImageItem(frame,axisOrder = 'row-major')
    img.show()
    vb.addItem(img)
    return img, vb


app,w = create_basic_app()
img, vb = create_video_player()

layout = QHBoxLayout()
control_layout = QVBoxLayout()
layout.addLayout(control_layout)
layout.addWidget(vb)

fig = create_fig()
fig.setTitle('rPPG')
addLabels(fig,'time','intensity','-','sec')
plt_r = plot(fig,np.arange(0,5),np.arange(0,5),[255,0,0])
plt_g = plot(fig,np.arange(0,5),np.arange(0,5),[0,255,0])
plt_b = plot(fig,np.arange(0,5),np.arange(0,5),[0,0,255])

fig_bpm = create_fig()
fig_bpm.setTitle('Frequency')
fig_bpm.setXRange(0,300)
addLabels(fig_bpm,'Frequency','intensity','-','BPM')
plt_bpm = plot(fig_bpm,np.arange(0,5),np.arange(0,5),[255,0,0])

layout.addWidget(fig)
layout.addWidget(fig_bpm)
timestamps = []
time_start = [0]

def update(load_frame,rPPG_extracter,rPPG_extracter_lukas,settings : Settings):
    bpm = 0    
    frame,should_stop,timestamp = load_frame() #frame_from_camera()
    dt = time.time()-time_start[0]
    fps = 1/(dt)
    
    time_start[0] = time.time()
    if len(timestamps) == 0:
        timestamps.append(0)
    else:
        timestamps.append(timestamps[-1] + dt)
        
    #print("Update")
    if should_stop:
        return
    rPPG = []
    if settings.use_flow: 
        rPPG_extracter = rPPG_extracter_lukas
        rPPG_extracter.crop_to_face_and_safe(frame)
        rPPG_extracter.track_Local_motion_lukas()
        rPPG_extracter.calc_ppg(frame)
        points = rPPG_extracter.points
        frame = cv2.circle(frame,(points[0,0,0],points[0,0,1]),5,(0,0,255),-1)
            
        rPPG = np.transpose(rPPG_extracter_lukas.rPPG)
    else:
        rPPG_extracter.measure_rPPG(frame,settings.use_classifier,settings.sub_roi) 
        rPPG = np.transpose(rPPG_extracter.rPPG)
    
        # Extract Pulse
    if rPPG.shape[1] > 10:
        if settings.use_resampling :
            t = np.arange(0,timestamps[-1],1/fs)
            
            rPPG_resampled= np.zeros((3,t.shape[0]))
            for col in [0,1,2]:
                rPPG_resampled[col] = np.interp(t,timestamps,rPPG[col])
            rPPG = rPPG_resampled
        num_frames = rPPG.shape[1]
        start = max([num_frames-100,0])
        t = np.arange(num_frames)/fs
        pulse = extract_pulse(rPPG,fftlength,fs)
        plt_bpm.setData(f,pulse) 
        plt_r.setData(t[start:num_frames],rPPG[0,start:num_frames])
        plt_g.setData(t[start:num_frames],rPPG[1,start:num_frames])
        plt_b.setData(t[start:num_frames],rPPG[2,start:num_frames])
        bpm = f[np.argmax(pulse)]
        fig_bpm.setTitle('Frequency : PR = ' + str(bpm) + ' BPM' )



    #print(fps)
    face = rPPG_extracter.prev_face
    if not settings.use_flow:
        if settings.show_cropped:
            frame = rPPG_extracter.frame_cropped
            if len(settings.sub_roi) > 0:
                sr = rPPG_extracter.sub_roi_rect
                draw_rect(frame,sr)
        else:
            try :
                draw_rect(frame,face)
                if len(settings.sub_roi) > 0:
                    sr = rPPG_extracter.sub_roi_rect
                    sr[0] += face[0]
                    sr[1] += face[1]
                    draw_rect(frame,sr)
            except Exception:
                print("no face")
#    write_text(frame,"bpm : " + '{0:.2f}'.format(bpm),(0,100))            
    write_text(frame,"fps : " + '{0:.2f}'.format(fps),(0,50))
    frame,_ = pg.makeARGB(frame,None,None,None,False)
    #cv2.imshow("images", np.hstack([frame]))
    img.setImage(frame)

timer = QtCore.QTimer()
timer.start(10)

def setup_update_loop(load_frame,timer,settings):
    # = "C:\\Users\\marti\\Downloads\\Data\\translation"
    try:  timer.timeout.disconnect() 
    except Exception: pass
    rPPG_extracter = rPPG_Extracter()
    rPPG_extracter_lukas = rPPG_Lukas_Extracter()
    update_fun = lambda : update(load_frame,rPPG_extracter,rPPG_extracter_lukas,settings)
    timer.timeout.connect(update_fun)

def setup_loaded_image_sequence(data_path,timer,settings):
    vl = VideoLoader(data_path)
    
    setup_update_loop(vl.load_frame,timer,settings)

def setup_webcam(timer,settings):
    camera = cv2.VideoCapture(0)
      #camera.set(3, 1280)
    #camera.set(4, 720)
    settings.use_resampling = True
    def frame_from_camera():
        _,frame = camera.read()
        return frame,False,camera.get(cv2.CAP_PROP_POS_MSEC)
        
    setup_update_loop(frame_from_camera,timer,settings)

def setup_video(data_path,timer,settings):
    vi_cap = cv2.VideoCapture(data_path)
    #settings.use_resampling = True
    def frame_from_video():
        _,frame = vi_cap.read()
        rows,cols = frame.shape[:2]

        #M = cv2.getRotationMatrix2D((cols/2,rows/2),270,1)
        #frame = cv2.warpAffine(frame,M,(cols,rows))
        frame = cv2.resize(frame,None,fx=0.5, fy=0.5, interpolation = cv2.INTER_CUBIC)
        return frame,False,0
        
    setup_update_loop(frame_from_video,timer,settings)

#settings = Settings()
if source.endswith('.mp4'):
    setup_video(source,timer,settings)
elif source == 'webcam':
    setup_webcam(timer,settings)
else :
    setup_loaded_image_sequence(source,timer,settings)


w.setLayout(layout)
execute_app(app,w)
camera.release()
cv2.destroyAllWindows() 

    


