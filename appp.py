# Simple camera app which can capture pictures from your webcam
# and display and store it in the current directory
from kivymd.app import MDApp
from kivy.lang.builder import Builder
from kivy.uix.boxlayout import BoxLayout

import time

Builder.load_string(
    '''
<CameraClick>:
    orientation: 'vertical'
    Camera:
        id:camera
        resolution: (1920, 1080)
        play: False
    
    MDRectangleFlatButton:
        text: "Play"
        on_press: camera.play = not camera.play
        size_hint_y: None
        height: '48dp'
    
    MDRectangleFlatButton:
        text: 'Capture'
        size_hint_y: None
        height: '48dp'
        on_press: root.capture()
    
'''
)


class CameraClick(BoxLayout):
    def capture(self):
        camera = self.ids['camera']
        timeit = time.strftime("%Y%m%d_%H%M%S")
        camera.export_to_png("IMG_{}.png".format(timeit))
        print("Captured")


class TestCamera(MDApp):
    def build(self):
        return CameraClick()


TestCamera().run()


