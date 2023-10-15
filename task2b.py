#!python3

import config
import helper

from task2a import retrieve as retrieve_task2a

if __name__ == '__main__':
  while True:
    inp = helper.get_user_input('img_id,K', len(config.DATASET), max(config.DATASET.y))
    inp['feat_space'] = 'resnet_softmax'
    retrieve_task2a(**inp)
  
