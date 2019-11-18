import requests
from PIL import Image
from io import BytesIO

def url2Img(url):
    response = requests.get(url)
    image = Image.open(BytesIO(response.content))
    return image

if __name__ == '__main__':
    img = url2Img('https://bdfile.bluemoon.com.cn/group2/M00/0A/BA/wKg_HlwzY1SAIdXDAAFyo-ZOLKQ399.jpg')
    img.show()