
# IBM Code Model Asset Exchange: Image Segmentation

This repository contains code to instantiate and deploy an image segmentation Deep Learning model in a Docker container as an API web service. This model takes an image file as an input and returns a segmentation map containing a predicted class for each pixel in the input image.

This repository contains 2 models trained on PASCAL VOC 2012. One model is trained using the xception architecture and produces very accurate results but takes a few seconds to run and the other model is trained on MobileNetV2 and is faster but less accurate. You can specify which model you wish to use when you start the Docker image. See below for more details.

The segmentation map returns an integer between 0 and 20 that corresponds to one of the labels below for each pixel in the input image. The first nested array corresponds to the top row of pixels in the image and the first element in that
array corresponds to the pixel at the top left hand corner of the image. 

**NOTE:** the image will be resized and the segmentation map refers to pixels in the resized image, not the original input image.

| Id | Label       | Id | Label       | Id | Label       |
|----|-------------|----|-------------|----|-------------|
| 0  | background  | 7  | car         | 14 | motorbike   |
| 1  | aeroplane   | 8  | cat         | 15 | person      |
| 2  | bicycle     | 9  | chair       | 16 | pottedplant |
| 3  | bird        | 10 | cow         | 17 | sheep       |
| 4  | boat        | 11 | diningtable | 18 | sofa        |
| 5  | bottle      | 12 | dog         | 19 | train       |
| 6  | bus         | 13 | horse       | 20 | tv          |


The model files are hosted on IBM Cloud Object Storage. The code in this repository deploys the model as a web service in a Docker container. This repository was developed as part of the [IBM Code Model Asset Exchange](https://developer.ibm.com/code/exchanges/models/), where other common models have been pretrained and available for download.



## Prerequisites
* `docker`: The [Docker](https://www.docker.com/) command-line iterface. Follow the [installations instructions](https://docs.docker.com/install/) for your system.


## Steps

To build and deploy the Deep Learning model to a REST API using Docker, follow these steps:

1. [Build the Model](#1-build-the-model)
2. [Deploy the Model](#2-deploy-the-model)
3. [Use the Model](#3-use-the-model)
4. [Development](#4-development)


### 1. Build the Model

Clone the `MAX-Image-Segmenter` repository locally. In a terminal, run the following command or [download](https://github.com/justinmccoy/MAX-Image-Segmenter/archive/master.zip) and extract this repo.

```
$ git clone https://github.com/justinmccoy/MAX-Image-Segmenter.git
```

Change directory into the repository base folder: 

```
$ cd MAX-Image-Segmenter
```

To build the docker image locally, run:

```
$ docker build -t max-image-segmenter .
```

All required model assets will be downloaded during the build process. _Note_ that currently this docker image is CPU only (we will add support for GPU images later).


### 2. Deploy the Model

To run the docker image, which automatically starts the model serving API, run:

```
$ docker run -it -p 5000:5000 max-image-segmenter
```

If you would like to specify what model or image size to load into the model, use -e flags to pass the API environmental variables:

```
$ docker run -it -e MODEL_TYPE='mobile' -e IMAGE_SIZE=333 -p 5000:5000 max-image-segmenter
```

_Note_ extra parameter info:
* Model types available: 'mobile', 'full' (default: mobile)
* Image size range: 16 to 1024 pixels (default: 513)

_Note_ that the image size parameter controls to what size the image will be resized to before it is processed by the model. Smaller images run faster but generate less accurate segmentation maps. 


### 3. Use the Model

The API server automatically generates an interactive Swagger documentation page. Go to `http://localhost:5000` to load it. From there you can explore the API and also create test requests.

Use the `model/predict` endpoint to load a test image (you can use one of the test images from the `assets` folder) and get predicted segmentation map for the image from the API.

![pic](docs/swagger-screenshot.png "Swagger Screenshot")

You can also test it on the command line, for example:

```
$ curl -F "image=@assets/stc.jpg" -XPOST http://localhost:5000/model/predict
```

You should see a JSON response like that below:

```
{
  "status": "ok",
  "image_size": [
    256,
    128
  ],
  "seg_map": [
    [
      0,
      0,
      0,
      ...,
      15,
      15,
      15,
      ...,
      0,
      0,
      0
    ],
    ...,
    ...,
    ...,
    [
      0,
      0,
      0,
      ...,
      15,
      15,
      15,
      ...,
      0,
      0,
      0
    ]
  ]
}
```


### 4. Development

To run the Flask API app in debug mode, edit `config.py` to set `DEBUG = True` under the application settings. You will then need to rebuild the Docker image (see [step 1](#1-build-the-model)).

**Taking a closer look at the code**

Our implementation is using Python Flask to front the deep learning model as a REST API, defining the endpoints and hosting the application as a web service.  Bundled within the Python web service and `/predict` API is an application that loads the trained deep learning image segmentation model using Tensorflow for Python, and wraps the model with some helper methods to simplify prediction when called from our Flask application.

Flask Web Service exposing two HTTP endpoints

```http
POST /model/predict
GET /model/metadata
```

**Let's dig into the code a bit:**

* **How is the container built?** _[(see step 1)](#1-build-the-model)_

  [app.py](https://github.com/justinmccoy/MAX-Image-Segmenter/blob/master/app.py) and the Deep Learning Model are copied into the container during build.
  >```docker
  >RUN wget -nv http://max-assets.s3-api.us-geo.objectstorage.softlayer.net/deeplab/deeplabv3_pascal_trainval_2018_01_04.tar.gz && \
  >mv deeplabv3_pascal_trainval_2018_01_04.tar.gz /workspace/assets/deeplabv3_pascal_trainval_2018_01_04.tar.gz
  >
  >COPY . /workspace
  >```

* **What application is started when loading the container?** 

  The [Docker File](https://github.com/justinmccoy/MAX-Image-Segmenter/blob/master/Dockerfile) shows where the pretrained model is downloaded from, how the application is packaged, and what is started.
  >```Docker
  ># Starts the Deep Learning Flask API for Image Segmentation
  >CMD python /workspace/app.py
  >```

  /workspace/app.py is a Python Flask app exposing 2 endpoints `POST /model/predict` and `GET /model/metadata`

  /workspace/app.py calls [api/model.py](https://github.com/justinmccoy/MAX-Image-Segmenter/blob/master/api/model.py) where the `/model/predict` endpoint is defined.

* **What's happening when the `/predict` URI is invoked?**
  >```Python
  > @api.route('/predict')
  > class Predict(Resource):
  >  model_wrapper = ModelWrapper()
  >
  >  @api.doc('predict')
  >  @api.expect(image_parser)
  >  @api.marshal_with(predict_response)
  >  def post(self):
  >      """Make a prediction given input data"""
  >      result = {'status': 'error'}
  >
  >      args = image_parser.parse_args()
  >      image_data = args['image'].read()
  >      image = read_image(image_data)
  >
  >      resized_im, seg_map = self.model_wrapper.predict(image)
  >
  >      result['image_size'] = resized_im.size
  >
  >      result['seg_map'] = seg_map
  >      result['status'] = 'ok'
  >      return result
  >```

  Above you see the `self.model_wrapper.predict(image)` is called, passing the image from the HTTP request to the `predict` method, where a segmentation map is returned, along side a resized version of the original image.  `predict` creates a new object `class DeepLabModel(object):` where the image segmentation model is loaded, and inference is run on the input image.

* **How is the model being loaded?**

  When `class DeepLabModel(object):` is instantiated the Deep Learning model (graph) is loaded into a TensorFlow Session as a TensorFlow Graph.

  >```Python
  >"""Creates and loads pre-trained deeplab model."""
  >      self.graph = tf.Graph()
  >
  >      graph_def = None
  >      # Extract frozen graph from tar archive.
  >      tar_file = tarfile.open(tarball_path)
  >      for tar_info in tar_file.getmembers():
  >          if self.FROZEN_GRAPH_NAME in os.path.basename(tar_info.name):
  >              file_handle = tar_file.extractfile(tar_info)
  >              graph_def = tf.GraphDef.FromString(file_handle.read())
  >              break
  >
  >      tar_file.close()
  >
  >      with self.graph.as_default():
  >          tf.import_graph_def(graph_def, name='')
  >
  >     self.sess = tf.Session(graph=self.graph)
  >


* **How do you call the model?**

  With the model loaded into a TensorFlow session, invoking it is the same as if you were testing the model after training in a Jupyter Notebook:
  >```Python
  > batch_seg_map = self.sess.run(
  >          self.OUTPUT_TENSOR_NAME,
  >          feed_dict={self.INPUT_TENSOR_NAME: [np.asarray(resized_image)]})
  > seg_map = batch_seg_map[0]
  >```



## References

* _Haozhi Qi, Zheng Zhang, Bin Xiao, Han Hu, Bowen Cheng, Yichen Wei, Jifeng Dai_, [Deformable Convolutional Networks -- COCO Detection and Segmentation Challenge 2017 Entry](http://presentations.cocodataset.org/COCO17-Detect-MSRA.pdf). ICCV COCO Challenge
    Workshop, 2017.

* _Mark Everingham, S. M. Ali Eslami, Luc Van Gool, Christopher K. I. Williams, John M. Winn, Andrew Zisserman_, [The Pascal Visual Object Classes Challenge: A Retrospective](http://host.robots.ox.ac.uk/pascal/VOC/voc2012/). IJCV, 2014.

* _Tsung-Yi Lin, Michael Maire, Serge Belongie, Lubomir Bourdev, Ross Girshick, James Hays, Pietro Perona, Deva Ramanan, C. Lawrence Zitnick, Piotr Dollar_, [Microsoft COCO: Common Objects in Context](http://cocodataset.org/). In the Proc. of ECCV, 2014.


## Licenses

| Component | License | Link  |
| ------------- | --------  | -------- |
| This repository | [Apache 2.0](https://www.apache.org/licenses/LICENSE-2.0) | [LICENSE](LICENSE) |
| Model Code (3rd party) | [Apache 2.0](https://www.apache.org/licenses/LICENSE-2.0) | [TensorFlow Models Repository](https://github.com/tensorflow/models/blob/master/LICENSE) |
| Model Weights | [Apache 2.0](https://www.apache.org/licenses/LICENSE-2.0) | [TensorFlow Models Repository](https://github.com/tensorflow/models/blob/master/LICENSE) |
| Test Assets | [Apache 2.0](https://www.apache.org/licenses/LICENSE-2.0) | [Asset README](assets/README.md)


, or use [PlayWithDocker](https://labs.play-with-docker.com/) online
* [IBM Cloud Account](https://cloud.ibm.com)
* *(Optional)* [Promo Code for Free Kubernetes Clusters](https://ibm.biz/promo-code)
* *(Optional)* IBM Cloud CLI [Installation Instructions](https://console.bluemix.net/docs/cli/reference/ibmcloud/cli_docker.html#using-ibm-cloud-developer-tools-from-a-docker-container)
* *(Optional)* IBM Cloud Kubernetes Service [Installation Instructions](https://cdeploymentloud.ibm.com/containers-kubernetes/catalog/cluster)