import streamlit as st

from itertools import islice

import deepstack.core as ds
from PIL import Image, ImageDraw
import pprint
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import av
import glob
from random import choice


with open('names.txt') as handle:
    names = [line.strip() for line in handle]


IP = '192.168.0.211'
PORT = '5000'
API_KEY = "auth_key"
TIMEOUT = 20 # Default is 10

NEED_REGISTER = False

PATHS = sorted(glob.glob('/videos/*/*.MP4'))

VID_NUM = st.sidebar.slider('Video Number', min_value = 0, max_value = len(PATHS), value = 0, key='path_num')

MODEL = st.sidebar.selectbox('Model', ['face_lite', 'face'])

MIN_CONFIDENCE = st.sidebar.slider('Min Condifence', min_value = 0.0, max_value = 1.0, step = 0.05,  value = 0.2, key='min_con')



DSFACE = ds.DeepstackObject(IP, PORT, API_KEY, TIMEOUT, custom_model=MODEL, min_confidence = MIN_CONFIDENCE)
#DSFACE = ds.DeepstackFace(IP, PORT, API_KEY, TIMEOUT)

FILE_PATH = PATHS[VID_NUM]

st.write(FILE_PATH)



def image_iter(files):
    
    for f in files:
        with av.open(f) as container:
            stream = container.streams.video[0]
            for num, frame in enumerate(container.decode(stream)):
                
                yield f, num, frame.to_image()


                
class Manager(object):
    
    def __init__(self, image_stream):
        
        self.image_stream = image_stream
        self.frame = None
        
    @staticmethod
    def from_local_video(path):
        
        img_iter = image_iter([path])
        img_only = map(lambda x: x[-1], img_iter)
        return Manager(img_only)
        
    def advance(self, frames = 1):
        
        for _ in range(frames):
            self.frame = next(image_stream)
            
        return self.frame
    
    def detect(self): 
        
        for res in DSFACE.detect(self.frame):
            yield res
            
    def play(self):
        
        while True:
            yield self.advance()
        
        
    #def draw_boxes(self, image = None):
        
def detect_faces(pil_img, tmp_path = '/tmp/tmp.jpg'):
    
    pil_img.save(tmp_path)
    
    for res in DSFACE.detect(open(tmp_path, 'rb')):
        yield res
    
def detect2array(detect):
    return np.array([detect['x_min'], detect['y_min'], detect['x_max'], detect['y_max']])


def detect2crop(detect):
    return [detect['x_min'], detect['y_min'], detect['x_max'], detect['y_max']]

def detect2rectangle(detect):
    return [(detect['x_min'], detect['y_min']), (detect['x_max'], detect['y_max'])]

    
    
MIN_COUNT = st.sidebar.slider('Min Frames', min_value = 1, max_value = 10, value = 3)
PIXEL_DIST = st.sidebar.slider('PixelDist', min_value = 0, max_value = 500, step = 5,  value = 200, key='pixel')
LOST_FRAMES = st.sidebar.slider('Lost Frames', min_value = 1, max_value = 10, step = 1,  value = 5, key='penalty')



class NaiveNode(object):
    
    def __init__(self, box, place):
        self.box = box
        self.count = 0
        self.lost = 0
        self.place = place
        self.name = choice(names)
        
        self.imgs = []
        
        self.name_button = None
        self.keep_button = None
        
    @staticmethod
    def from_detect(detect, place):
        
        return NaiveNode(detect2crop(detect), place)
    
    def score(self, detect):
        
        return np.sum(np.sqrt((np.array(self.box)-detect2array(detect))**2))
    
    def step(self, detect):
        self.box = [int(i) for i in (np.array(self.box) + detect2array(detect))/2]
        self.count += 1
        self.lost = 0
        
    def back(self):
        self.lost += 1
    
    def keep(self):
        return self.lost < LOST_FRAMES

    @property
    def rectangle(self):
        b = self.box
        return [(b[0], b[1]), (b[2], b[3])]
    
    def should_crop(self, min_count = MIN_COUNT):
        return self.count >= min_count
        
        
    def render(self, img):
        
        if type(self.place) is not tuple:
            self.place = next(self.place)
        
        self.imgs.append(img.copy().crop(self.box))
        self.place[0].image(self.imgs[-1])

        if self.name_button is None:
            self.name_button = st.text_input('Name', value = self.name, key=self.name+'name')
            self.keep_button = st.checkbox("Keep?", key = self.name+'keep')
        self.place[1].write(f'Frames: {self.count}')
           



    
class NodeHolder(object):
    
    def __init__(self, nodes):
        
        self.all_nodes = []
        self.nodes = nodes
        
    def check_detect(self, detect):
        
        scores = [(node.score(detect), num,  node) for num, node in enumerate(self.nodes)]
        return min(scores, key = lambda x: x[0])    
            
    def get_node_or_create(self, detect, cutoff = 1e-8):
        
        
        if len(self.nodes) == 0:
            self.nodes.append(NaiveNode.from_detect(detect, FACE_PLACES))
            self.all_nodes.append(self.nodes[-1])
        else:
            score, num, node = self.check_detect(detect)
            if score < PIXEL_DIST:
                node.step(detect)
                return node
            else:
                node.back()
                self.nodes.append(NaiveNode.from_detect(detect, FACE_PLACES))
                self.all_nodes.append(self.nodes[-1])
        return self.nodes[-1]
    
    
    def process_img(self, img):
        
        found = []
        draw_img = ImageDraw.Draw(img)
        faces = list(detect_faces(img))
        for detect in faces:
            node = self.get_node_or_create(detect)
            draw_img.rectangle(node.rectangle)
        [node.back() for node in self.nodes]
        
        self.nodes = [node for node in self.nodes if node.keep()]
        return draw_img
            
    
    def crop_nodes(self, img):
        
        for node in self.nodes:
            if node.should_crop():
                node.render(img)
                
        
        
    
    
st.title('Sea Lion Face ID')


image_placeholder = st.empty()


def gen_empty(mx = 20):
    for m in range(mx):
        pic, fm = st.columns(2)
        yield pic.empty(), fm.empty()
        st.write('---------------------------------------')


FACE_PLACES = gen_empty()


#ctr = Manager.from_local_video(FILE_PATH)
stream = islice(image_iter([FILE_PATH]), 50000)

_, _, img = next(stream)
image_placeholder.image(img)    



if st.button('Start'):
    holder = NodeHolder([])
    with st.form(key = 'form') as form:
        
        for p, frame, img in stream:

            draw_img = holder.process_img(img)

            holder.crop_nodes(img)

            image_placeholder.image(img)
        
        
        
        st.write('Done')
        st.form_submit_button()
        
    for node in holder.all_nodes:
        if node.keep_button:
            st.write(f'Would write {node.name_button} {len(node.imgs)}')
