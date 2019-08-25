# Transfer Learning

...

## Datasets

1. 3,000 images of dogs 🐶 vs cats 🐱 vs pandas 🐼, a 1,000 each

Dog             |  Cat             |  Panda
:-------------------------:|:-------------------------:|:-------------------------:
![](docs/dog.jpg)  | ![](docs/cat.jpg)  | ![](docs/panda.jpg)

See [here]() for further details

## Technologies

* [keras](https://keras.io)
* [tensorflow](https://www.tensorflow.org/)
* [pillow](https://pillow.readthedocs.io/)
* [scikit-learn](https://scikit-learn.org/)
  
## Deployment

...

```
$ python src/extract_features.py --dataset datasets/animals/images --output datasets/animals/hdf5/features.hdf5
[INFO] loading images...
[INFO] loading network...
Downloading data from https://github.com/fchollet/deep-learning-models/releases/download/v0.1/vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5
58892288/58889256 [==============================] - 5s 0us/step
...
```

```
$ python src/train_model.py --db datasets/animals/hdf5/features.hdf5 --model models/animals.cpickle
...
```

## Resources

* Deep Learning for Computer Vision with Python by Dr. Adrian Rosebrock: https://www.pyimagesearch.com/deep-learning-computer-vision-python-book/