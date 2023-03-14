from sklearn.model_selection import train_test_split
from selenium import webdriver
from selenium.webdriver.common.by import By
from time import sleep
from PIL import Image
import numpy as np
import requests
import os
import io


#initialize dir variables
filedir = os.getcwd()

def init_dataset():
    pass

def get_data(data_len=5000, ow=False):

    #init variables
    inputs = []
    outputs = []

    try:
        if not ow:
            inputs, outputs = np.load(f"{filedir}/dataset/processed_data.npy")
        else:
            raise FileNotFoundError
        
    except FileNotFoundError:
        inputs, outputs = scrape_images(data_len)
        inputs = np.asarray(inputs)
        outputs = np.asarray(outputs)
        np.save(f"{filedir}/dataset/processed_data.npy", (np.array([inputs, outputs])))

     #prepare data for neural network
    inputs_formatted = inputs / 255.0 #divide image data by 255, so that the neural nets gets all data between 0 and 1
    outputs_formatted = outputs / 255.0
    inputs_train, inputs_test, outputs_train, outputs_test = train_test_split(inputs_formatted, outputs_formatted, test_size=0.3) #split data between testing and training data

    #return data
    return (inputs_train, inputs_test, outputs_train, outputs_test)

def get_images(url, delay=1):
    
    image = Image.open(io.BytesIO(requests.get(url).content))
    input_imgs = []
    output_imgs = []
    
    width, height = image.size
    add = 0

    for x in range(0, width - 128, 128):
        for y in range(0, height - 128, 128):
            output_img = image.crop((x, y, x + 128, y + 128))
            input_img = output_img.resize((64, 64))
            output_imgs.append(output_img)
            input_imgs.append(input_img)
            add += 1

    sleep(delay)
    return add, input_imgs, output_imgs

def scroll(driver):
        driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")

def scrape_images(data_len=5000, thumbnail_class="Q4LuWd", image_class="n3VNCb", page="http://bitly.ws/BBX5"): #search_tag is the class name in html
    
    #init driver
    PATH = "C:\\Users\\filip\\OneDrive\\Desktop\\Other\\chromdriver.exe"
    driver = webdriver.Chrome(PATH)
    driver.get(page)
    sleep(5) #user needs to press accept

    #init variables
    inputs = []
    outputs = []

    thumbnails = []
    delay = 0.1

    thumbnail_c = 0
    data_c = 0


    #scraping loop
    while data_c <= data_len:
        try:
            if len(thumbnails) == 0:
                raise Exception
            
            for thumbnail in thumbnails: #click on each thumbnail to get full quality photo
                try:
                    thumbnail.click() 
                    urls = driver.find_elements(By.CLASS_NAME, image_class)
                    for url in urls:
                        url = url.get_attribute("src")
                        if "http" in url:
                            add, input_imgs, output_imgs = get_images(url, delay=delay)
                            for i1 in range(len(input_imgs)): #get all subimages from image
                                inputs.append(input_imgs[i1])
                                outputs.append(output_imgs[i1])
                            data_c += add
                            thumbnail_c += 1
                except Exception as e:
                    print(e)
                    continue


        except Exception as e:
            print(e)

            thumbnails = driver.find_elements(By.CLASS_NAME, thumbnail_class)[thumbnail_c:]

            #click on show more
            try: 
                driver.find_element(By.CLASS_NAME, "r0zKGf").click()
            except Exception:
                try:
                    driver.find_element(By.CLASS_NAME, "mye4qd").click()
                except Exception:
                    pass
            
            scroll(driver)
        
    driver.quit()
    return inputs, outputs

if __name__ == "__main__":
    print(get_data(100, ow=True))