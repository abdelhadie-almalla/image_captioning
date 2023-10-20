This image captioning project is the code for the paper:

Al-Malla MA, Jafar A, Ghneim N. Image captioning model using attention and object features to mimic human image understanding. Journal of Big Data. 2022 Dec;9(1):1-6.
https://doi.org/10.1186/s40537-022-00571-w.

Please cite the paper if you use this code.

The input is an image and the output is a sentence that describes the input image. The solution uses attention with deep learning.

You need to download yolov4.weights from https://wiki.loliot.net/docs/lang/python/libraries/yolov4/python-yolov4-about

1. image_captioning_version_16.1--feature_concatination--boundingboxes_v2.py: training, validation, and testing.

The main file. Contains the code for training the model, and then runs the model to predict the captions on the validation set.
The used dataset is MS COCO. MS COCO only has training and validation splits, so in this project the MS COCO train split is used to train and validate the model, with 20% for the validation split and 80% for the training set.
The MS COCO validation set is used as the testing set, because the testing set available on the MS COCO website does not contain labels.
This file dumps a file called "results.json" that contains an array of json objects. The file has this format:

[{
"image_id": int, "caption": str,
}]

which the evaluation code expects. This is the default format for MS COCO.
This file also dumps the trained model in two files named "my_model.index" and "my_model.data-00000-of-00001", and a file called "checkpoint" in the tensorflow format, and a file "max_length.txt". The script loads the pretrained model in case it exists, this can be changed in the model parameters. It also dumps "tokenizer.pickle" to save the tokenizer.

This file was started from the tensorflow example available at https://www.tensorflow.org/tutorials/text/image_captioning

Example use:
python image_captioning_version_16.1--feature_concatination--boundingboxes_v2.py

Note: you need to be at the directory of the file using a "cd" command before executing this file. This file is written in Python 3

2. evaluate.py: evaluation of the result file. Uses the evaluation code available from the official MS COCO github repository at
https://github.com/tylin/coco-caption
I have slightly edited this file to accept the file names as a command line parameter.
I have slightly edited the evaluation library, to disable caching for the SPICE evaluation metric. Caching speeds up execution but needs so much disk space that it produces errors if such space is not available. This edit is documented in the file "spice.py".
Takes as parameters the result file and the original caption file.
This file does not need to be executed from a specific directory.

Example use:
python evaluate.py results.json captions_train2014.json

This file is written in Python 2
Obtained from: https://github.com/tylin/coco-caption/blob/master/cocoEvalCapDemo.ipynb

3. model.py

Contains the model and hyper parameters.

4. caption.py

Describes an image. Uses duplicate code from image_captioning_version_16.1--feature_concatination--boundingboxes_v2.py to avoid splitting the original code to multiple files. Example use:
python caption.py myimage.jpg

Note: you need to be at the directory of the file using a "cd" command before executing this file. 

--data--
this code is configured to accept flickr8k, flickr30k or the MSCOCO datasets

annotations: mscoco annotation directory
annotations_flickr8k: flickr8k annotaion directory
annotations_flickr30k: flickr30k annotaion directory
flickr8k: flickr8k images
flickr30k: flickr30k images
