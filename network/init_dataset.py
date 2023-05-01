from tensorflow.data import Dataset
from npy_append_array import NpyAppendArray
from selenium import webdriver
from selenium.webdriver.common.by import By
from tensorflow import keras
from time import sleep
from PIL import Image
import numpy as np
import requests
import os
import io


#initialize dir variables
filedir = os.getcwd()

def get_data(data_len=5000, ow=False):

    #init variables
    inputs = []
    outputs = []

    try:
        if not ow:
            inputs = np.load(f"{filedir}/dataset/processed_input_data.npy")
            outputs = np.load(f"{filedir}/dataset/processed_output_data.npy")
        else:
            raise FileNotFoundError
        
    except FileNotFoundError:
        inputs, outputs = scrape_images(data_len)
        inputs = np.array(inputs)
        outputs = np.array(outputs)

     #prepare data for neural network
    inputs = inputs / 255.0 #divide image data by 255, so that the neural nets gets all data between 0 and 1
    outputs = outputs / 255.0


    #convert data to tf dataset
    train_dataset = Dataset.from_tensor_slices((outputs, inputs))

    #return data
    return train_dataset


#class names can be different, in that case check what class names google images has using the html inspector and update the variable
def scrape_images(data_len=5000, filename=f"{filedir}/dataset/processed_", thumbnail_class="Q4LuWd", image_class="iPVvYb", pages=["https://www.google.com/search?q=popular+photo&source=lnms&tbm=isch&sa=X&ved=2ahUKEwjo29zFtd79AhXWi_0HHSZrCN4Q_AUoAXoECAIQAw&biw=2844&bih=1518&dpr=0.9", "https://www.google.com/search?q=faces&sxsrf=APwXEddEcQzRXvB_9zsVepWQnQZOpDw0Ig:1680807930842&source=lnms&tbm=isch&sa=X&ved=2ahUKEwj63py9-ZX-AhVTg_0HHXo3Bb0Q_AUoAXoECAIQAw&biw=2844&bih=1518&dpr=0.9", "https://www.google.com/search?q=landscape&tbm=isch&ved=2ahUKEwj2qNq--ZX-AhVRricCHWu2AFQQ2-cCegQIABAA&oq=landscape&gs_lcp=CgNpbWcQAzIHCAAQigUQQzIFCAAQgAQyBQgAEIAEMgUIABCABDIFCAAQgAQyBQgAEIAEMgUIABCABDIFCAAQgAQyBQgAEIAEMgUIABCABDoECCMQJzoHCCMQ6gIQJ1CEBVj0G2DmHmgBcAB4AoABf4gBvAiSAQQxMy4xmAEAoAEBqgELZ3dzLXdpei1pbWewAQrAAQE&sclient=img&ei=_RcvZLa2OdHcnsEP6-yCoAU&bih=1518&biw=2844", "https://www.google.com/search?q=city&tbm=isch&ved=2ahUKEwiL_-vI-ZX-AhXRnycCHQ9rAnAQ2-cCegQIABAA&oq=city&gs_lcp=CgNpbWcQAzIHCAAQigUQQzIHCAAQigUQQzIHCAAQigUQQzIHCAAQigUQQzIHCAAQigUQQzIHCAAQigUQQzIFCAAQgAQyBwgAEIoFEEMyBQgAEIAEMgcIABCKBRBDOgQIIxAnUOANWL0TYM8VaABwAHgAgAFQiAHbApIBATWYAQCgAQGqAQtnd3Mtd2l6LWltZ8ABAQ&sclient=img&ei=ExgvZMupDNG_nsEPj9aJgAc&bih=1518&biw=2844", "https://www.google.com/search?q=art&sxsrf=APwXEdcNpmhHDCtZKL0buT2kMV5KK1kkHQ:1680807999073&source=lnms&tbm=isch&sa=X&ved=2ahUKEwimquHd-ZX-AhVp_rsIHWVuDiYQ_AUoAXoECAIQAw&biw=2844&bih=1518&dpr=0.9"]): #search_tag is the class name in html
    
    #init driver
    PATH = "C:...\\chromdriver.exe" #has to be setup
    driver = webdriver.Chrome(PATH)
    driver.get(pages[0])
    sleep(.5)
    driver.find_element(By.CLASS_NAME, "Nc7WLe").click()
    sleep(.5)

    #init variables
    inputs = []
    outputs = []

    thumbnails = []
    delay = 0.1

    thumbnail_c = 0
    data_c = 0

    page_c = 0

    npaa_input = NpyAppendArray(filename + "input_data.npy", delete_if_exists=True)
    npaa_output = NpyAppendArray(filename + "output_data.npy", delete_if_exists=True)


    #scraping loop
    while data_c <= data_len:

        thumbnails = driver.find_elements(By.CLASS_NAME, thumbnail_class)[thumbnail_c:]

        if len(thumbnails) == 0:

            #click on show more
            thumbnail_c = 0
            try: 
                driver.find_element(By.CLASS_NAME, "r0zKGf").click()
            except Exception:
                try:
                    driver.find_element(By.CLASS_NAME, "mye4qd").click()
                except Exception:
                    if page_c >= len(pages):
                        return np.load(f"{filedir}/dataset/processed_input_data.npy"), np.load(f"{filedir}/dataset/processed_output_data.npy")
                    driver.get(pages[page_c])
                    page_c += 1

        
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

                        if data_c >= data_len:
                            driver.quit()
                            return np.load(f"{filedir}/dataset/processed_input_data.npy"), np.load(f"{filedir}/dataset/processed_output_data.npy")
                        
                        progress_bar = "|" + ("*" * int(data_c / data_len * 100)) + (" " * int(100 - data_c / data_len * 100)) + f"| {int(data_c / data_len * 100)}%"
                        print(progress_bar, end="\r", flush=True)
            except Exception as e:
                print(e)
                continue

            #if there is already more then 100 pieces of data, save files to save RAM
            if len(inputs) >= 1000:
                try:
                    npaa_input.append(np.array(inputs))
                    npaa_output.append(np.array(outputs))
                    print("\n\nappended to npy files\n\n")
                    inputs = []
                    outputs = []
                except Exception as e:
                    print(f"failed to upload data: {e}")
            
    npaa_input.append(np.array(inputs))
    npaa_output.append(np.array(outputs))
    driver.quit()
    return np.load(f"{filedir}/dataset/processed_input_data.npy"), np.load(f"{filedir}/dataset/processed_output_data.npy")

def get_images(url, delay=1):
    
    image = Image.open(io.BytesIO(requests.get(url).content)).convert("RGB")
    input_imgs = []
    output_imgs = []
    
    width, height = image.size
    add = 0

    #split images into tiles, which will be resized and saved to the input and output img lists
    for x in range(0, width - 128, 128):
        for y in range(0, height - 128, 128):
            output_img = image.crop((x, y, x + 128, y + 128))
            input_img = output_img.resize((64, 64))
            output_imgs.append(keras.utils.img_to_array(output_img))
            input_imgs.append(keras.utils.img_to_array(input_img))
            add += 1

    sleep(delay)
    return add, input_imgs, output_imgs

def scroll(driver):
        driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")

def test_get_imgs():
    print(get_images("https://images.google.com/images/branding/googlelogo/1x/googlelogo_light_color_272x92dp.png", delay=0))
    print(get_images("https://fastly.4sqi.net/img/user/32x32/4UFEZ5POXMAZVHGU.jpg", delay=100))

if __name__ == "__main__":
    get_data(10000, ow=True)