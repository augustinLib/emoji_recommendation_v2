from PIL import Image           
import random

def return_image(sentiment):
        
    if (sentiment == "짜증"):
        num = random.randrange(1,3)
        if num == 1:
            img = Image.open('./emoji/fear.jpeg')
            img.show()
        else:
            img = Image.open('./emoji/angry.png')
            img.show()

    elif (sentiment == "슬픔"):
        img = Image.open('./emoji/sad.jpeg')
        img.show()


    elif (sentiment == "당황"):
        img = Image.open('./emoji/surprised.jpeg')
        img.show()

    else:
        num = random.randrange(1,3)
        if num == 1:
            img = Image.open('./emoji/joy.png')
            img.show()
        else:
            img = Image.open('./emoji/love.jpeg')
            img.show()
