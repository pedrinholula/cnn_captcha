import random
import os
def create_sample (path = "./data/samples", size =10):
	### Create a random size sample with the images on the gived path;
	return random.choices(os.listdir(path),k=size)