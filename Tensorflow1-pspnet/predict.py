from PIL import Image

from pspnet import Pspnet

pspnet = Pspnet()

while True:
    img = input('Input image filename:')
    try:
        image = Image.open(img)
    except:         
        print('Open Error! Try again!')
        continue
    else:
        r_image = pspnet.detect_image(image)
        r_image.show()
