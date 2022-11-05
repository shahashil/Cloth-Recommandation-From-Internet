#https://www.youtube.com/watch?v=OQydrlSzxnE

from google_images_download.google_images_download import google_images_download

#instantiate the class
def download_images(keywords):
    response = google_images_download.googleimagesdownload()
    arguments = {"keywords":keywords,"limit":20,"print_urls":False}
    try:
        paths = response.download(arguments)
        return True
    except:
        return False


if __name__ == '__main__':
    download_images("red shirt")