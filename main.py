import cv2
import numpy as np
import json

def print_name(img, names, cup_centers = [(421,852), (782,780)], angles = [3, 13], show = True):

    img = img / 255.0
    img_size = img.shape
    font = cv2.FONT_HERSHEY_SIMPLEX

    out = img.copy()

    for name, cup_center, angle in zip(names, cup_centers, angles):

        text_img = np.ones(out.shape)
        text_size =cv2.getTextSize(name, font, 0.9, cv2.LINE_AA)[0]
        text_start = (int(img_size[0]/2-text_size[0]/2), int(img_size[1]/2-text_size[1]/2))
        
        # text in the middle 
        text_img = cv2.putText(text_img,name,text_start, font, 0.9, (0.35, 0.35, 0.35), 2, cv2.LINE_AA)

        # rotate text
        text_img = rotate_img(text_img,angle)

        # move text from the middle to the cup
        text_start_cup = (-int(img_size[0]/2 - cup_center[0]),-int(img_size[1]/2 - cup_center[1]))
        text_img = translate_img(text_img, text_start_cup[0], text_start_cup[1])

        # merge text layer with the image
        out = np.multiply(out,text_img)

    # show image
    if show:
        cv2.imshow("out",out)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    # save image
    out = cv2.convertScaleAbs(out, alpha=(255.0))
    cv2.imwrite("&".join(names)+"'s cup.jpg",(out))

def rotate_img(img, angle):
    # get rotation matrix
    rot_mat = cv2.getRotationMatrix2D((int(img.shape[0]/2), int(img.shape[1]/2)), angle, 1.0)
    # rotate img
    img = cv2.warpAffine(img, rot_mat, img.shape[1::-1], flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT,borderValue=(1,1,1))
    return img

def translate_img(img, x, y):
    # get translation matrix
    tra_mat = np.float32([[1,0,x],[0,1,y]])
    # move the image
    img = cv2.warpAffine(img,tra_mat,img.shape[1::-1], borderMode=cv2.BORDER_CONSTANT,borderValue=(1,1,1))
    return img

if __name__ == "__main__":
    img = cv2.imread("img.jpg", cv2.CV_32F)
    with open('names.json') as names_file:
        data = json.load(names_file)[:4]
        for name1,name2 in zip(data[0::2], data[1::2]):
            print_name(img, ["Amadeus",name2], show = False)