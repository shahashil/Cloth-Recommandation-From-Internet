# Cloth-Recommandation-From-Internet
## To Recommend Pants/Jeans from Given Tshirts/Shirt from Internet.

* Firstly Train the Images in your Dataset in train_resnet.py and extract all the features.

* In the Main.py we have to describe the Current Topwear we want to get recommendation for. 
That description will be further used to get Google Images of the model wearing that. 

* To get the Full body pic of model we will use MediaPipe Library in pose.py
to extract the Human Pose points and filter out the Unwanted one.
<img src="https://raw.githubusercontent.com/shahashil/Cloth-Recommandation-From-Internet/main/download_images_from_internet.PNG" >

* Again using Mediapipe we will crop the images from hips to ankle so as to 
get the Focused Feature i.e. Pants. using cloth_extraction.py
<img src="https://raw.githubusercontent.com/shahashil/Cloth-Recommandation-From-Internet/main/extracted_cloth.PNG" >

* The Pants will then be Fed to Recommendation model that has been trained on Resnet50 model, having around 2048 features for each Image.
The model will map to the closest available Pants in your Inventory and return
the image name.




