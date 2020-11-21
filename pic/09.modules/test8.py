
# import urllib.request
# url= 'http://202.112.85.96/python/ref'
# response = urllib.request.urlopen(url)
# html = response.read()
# #print(html)
# from bs4 import BeautifulSoup as bs
# soup = bs(html, 'html.parser')
# links = soup.findAll('a')
# # print(links)
# for a  in   links:
#     print(url+a['href'])
#     if  "jpg" in a[ 'href']:
#     urllib.request.urlretrieve(url+a['href'], a['href'])

# import matplotlib.pyplot as plt
# img = plt.imread('BNULogo.jpg')
# img_s= img[::2,::2]
# plt.imsave("logo_sml.png", img_s)
# img_c= img[   400:600 ,400:600 ]
# plt.imsave("logo_crop.png", img_c)
# import numpy as np
# import matplotlib.pyplot as plt
# comb = np.zeros([992,1280,3])
# fname= "opo0907h"
# for i in range(3):
#     img = plt.imread(fname+"_" +str(i)+'.jpg')
#     print(i,img.shape)
#     comb[:,:,i] = img/255.
# plt.imshow(comb)
# plt.show()
# plt.imsave(fname+"_comb.jpg", comb)



# import numpy as np
# import matplotlib.pyplot as plt
# #读取原始图像
# img=plt.imread('IMG_1556_c.jpg')#returnRGB
# #获取图像行和列
# rows,cols=img.shape[:2]
# B=np.sqrt(img[:,:,2])*12
# B[B>255]=255
# G=img[:,:,1];R=img[:,:,0]
# img_f=np.stack((R,G,B),axis=2)
# img_f=img_f.astype('uint8')
# #~print(img_f[:3,:3,:])

# plt.imsave('IMG_1556_r.png', img_f)

# import os
# import numpy as np
# import matplotlib.pyplot as plt
# from PIL import Image
# im=Image.open(os.path.join(str(os.getcwd())+'/'+"BNULogo.jpg"))
# print(im.size)
# im2=im.resize((300,300),Image.ANTIALIAS)
# im2.save("bnu_logo_scaled.jpg",quality=90)
# im3=im.crop((400,400,600,600))
# im3.save("bnu_logo_croped.jpg")
# im4=im.rotate(45)
# plt.imshow(im4)
# plt.show()


from PIL import Image, ImageDraw
import pylab as plt
import random
nimg= Image.new('RGB',(60,30),'red')
draw = ImageDraw.Draw(nimg)
draw.text((10,10 ),'1 2 3')
plt.imshow(nimg)
plt.show()


