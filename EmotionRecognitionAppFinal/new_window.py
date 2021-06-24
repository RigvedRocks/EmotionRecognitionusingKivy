from kivy.uix.filechooser import FileChooserIconView
from kivy.uix.widget import Widget
from kivy.app import App
from kivy.lang import Builder
from kivy.uix.screenmanager import ScreenManager,Screen
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np


class FirstWindow(Screen):
     pass

class SecondWindow(Screen):
    def __init__(self, **kw):
         super().__init__(**kw)
     
    def recognition(self,filename):
        
            if(filename[0].endswith(('.png', '.jpg', '.jpeg'))):
                self.ids.img.source = filename[0]
                mymodel = load_model('my_model.h5')
                img = image.load_img(filename[0],target_size=(210,210,3))
                img = image.img_to_array(img)
                img = np.expand_dims(img,axis = 0)
                num = int(mymodel.predict_classes(img)[0])
                emotion = {0: 'Angry',1: 'Confused',2: 'Embarrased',3: 'Excited',4: 'Happy',5: 'Sad',6:'Scared'}
                prediction = emotion[num]
                self.ids["my_label"].text = "Predicted Emotion : " + prediction
                filename.clear()
            else:
                self.ids["my_label"].text = "Please upload an image file" 
                filename.clear()
         
class WindowManager(ScreenManager):
    pass

kv = Builder.load_file('new_window.kv')


class MyApp(App):
    def build(self):
        return kv

if __name__=="__main__":
    MyApp().run()