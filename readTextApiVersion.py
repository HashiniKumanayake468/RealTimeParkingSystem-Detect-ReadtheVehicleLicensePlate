

def visionApi():
        import os
        import sys
        import requests
        # If you are using a Jupyter notebook, uncomment the following line.
        # %matplotlib inline
        import matplotlib.pyplot as plt
        from matplotlib.patches import Rectangle
        from PIL import Image
        from io import BytesIO


        subscription_key = 'e9be342af8164e3d82d2af1059193221'
        endpoint ='https://eastus.api.cognitive.microsoft.com/'

        ocr_url = endpoint + "vision/v3.0/ocr"

        # Set image_url to the URL of an image that you want to analyze.
        # image_url = "https://upload.wikimedia.org/wikipedia/commons/thumb/a/af/" + \
        #     "Atomist_quote_from_Democritus.png/338px-Atomist_quote_from_Democritus.png"

        # headers = {'Ocp-Apim-Subscription-Key': subscription_key}
        # params = {'language': 'unk', 'detectOrientation': 'true'}
        # data = {'url': image_url}
        # response = requests.post(ocr_url, headers=headers, params=params, json=data)
        # response.raise_for_status()

        # analysis = response.json()


        image_path = "C:/tensorflow3/models/research/object_detection/contour.jpg"
        # Read the image into a byte array
        image_data = open(image_path, "rb").read()
        params = {'language': 'en'}
        # Set Content-Type to octet-stream
        headers = {'Ocp-Apim-Subscription-Key': subscription_key, 'Content-Type': 'application/octet-stream'}
        # put the byte array into your post request
        response = requests.post(ocr_url, headers=headers, params=params, data = image_data)

        analysis = response.json()

        # line_infos = [region["lines"] for region in analysis["regions"]]
        # word_infos = []
        # for line in line_infos:
        #     for word_metadata in line:
        #         for word_info in word_metadata["words"]:
        #             word_infos.append(word_info)
        # word_infos
        print(analysis)

visionApi()