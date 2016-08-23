Repo for [San Francisco Crime Kaggle challenge](https://www.kaggle.com/c/sf-crime). Predicting crime classifications from arrest data, principally geopositional.

This Kaggle competition was entered as part of the coursework requirement for The Applied Machine Learning module of the UCL MSc. in Computational Statistics and Machine Learning.

Team Bayesian Bandits: Alberto Martin, Chris Hart, Henrietta Forssen and Rikk Hill

The competition has now passed, as has the UCL module, so I'm making this repo public. We ended up in the top 6% of entrants for the competition. Our final method involved a fairly pedestrian Convolutional Neural Network, but this repo represents a much nicer set of tools for exploring and visualising the data.

Included is a utility for carrying out multiclass binomial regression on arbitrary panel data, and an abandoned effort to implement the perceptron algorithm for linear separation of arbitrary high-dimensional data.

---

I've started writing some helper functions for dealing with the data. At the moment
it's just data-related functions. You can call them in this sort of fashion:

```python
import helper.data as hd

# Get the training data
training = hd.get_training_data()
```

Run your scripts in a context where the pwd is the root directory (`sfcrime`). If you don't know how to do that, just put them in the root directory and run them. If you want to get fancy and write some modules, put the module directory in the root too (alongside `helper`).

Download the data from Kaggle and unzip it in the `data` directory, then run `setup.py` from the root directory.
This will create 80/20 split training and validation subsets in the data directory, as well as various other things that I've yet to cleverly come up with.
