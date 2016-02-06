Repo for San Francisco Crime Kaggle challenge

This should be private. If you can see it, and you're not Rikk, Chris, Henrietta or Alberto, then stop looking at it, or we'll tell on you.

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
