from kivy.app import App
from kivy.uix.widget import Widget
from kivy.uix.label import Label
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.button import Button
from kivy.uix.filechooser import FileChooserIconView
from kivy.properties import ObjectProperty
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np


class MyGrid(Widget):
    label = ObjectProperty(None)
    img = ObjectProperty(None)
    filechooser = ObjectProperty(None)
    def classify_image(self):
        img_path = self.ids['img'].source
        mymodel = load_model('my_model.h5')
        img = image.load_img(img_path,target_size=(210,210),color_mode='rgb')
        img = image.img_to_array(img)
        img = np.expand_dims(img,axis=0)
        num = int(mymodel.predict_classes(img)[0])
        emotion = {0: 'Angry',1: 'Confused',2: 'Embarrased',3: 'Excited',4: 'Happy',5: 'Sad',6:'Scared'}
        prediction = emotion[num]
        self.ids["label"].text = "Predicted Class : " + prediction
    
class MyApp(App):
    def build(self):
        return MyGrid()
    
        
if __name__=="__main__":
    app = MyApp()
    app.run()