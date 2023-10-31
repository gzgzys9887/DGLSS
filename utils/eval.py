#!/usr/bin/env python3
import numpy as np
import time


class SemanticEval:
  """ Semantic evaluation using numpy
  
  authors: Andres Milioto and Jens Behley
  """

  def __init__(self, n_classes, device=None, ignore=None):
    self.n_classes = n_classes
    assert (device == None)
    self.ignore = np.array(ignore, dtype=np.int64)
    self.include = np.array([n for n in range(self.n_classes) if n not in self.ignore], dtype=np.int64)

    print("[SEMANTIC EVAL] IGNORE: ", self.ignore)
    print("[SEMANTIC EVAL] INCLUDE: ", self.include)

    self.reset()
    self.eps = 1e-15

  def num_classes(self):
    return self.n_classes

  def reset(self):
    # iou 
    self.px_iou_conf_matrix = np.zeros((self.n_classes, self.n_classes), dtype=np.int64)
    
  ################################# IoU Semantic ##################################
  def addBatchSemIoU(self, x_sem, y_sem):
    # idxs are labels and predictions
    idxs = np.stack([x_sem, y_sem], axis=0)

    # make confusion matrix (cols = gt, rows = pred)
    np.add.at(self.px_iou_conf_matrix, tuple(idxs), 1)

  def getSemIoUStats(self):
    # clone to avoid modifying the real deal
    conf = self.px_iou_conf_matrix.copy().astype(np.double)
    # remove fp from confusion on the ignore classes predictions
    # points that were predicted of another class, but were ignore
    # (corresponds to zeroing the cols of those classes, since the predictions
    # go on the rows)
    conf[:, self.ignore] = 0

    # get the clean stats
    tp = conf.diagonal()
    fp = conf.sum(axis=1) - tp
    fn = conf.sum(axis=0) - tp
    return tp, fp, fn

  def getSemIoU(self):
    tp, fp, fn = self.getSemIoUStats()
    include_things = [i for i in self.include if i in [1,2,3,4,5,6]]
    include_stuff = [i for i in self.include if i in [7,8,9,10]]
    # print(f"tp={tp}")
    # print(f"fp={fp}")
    # print(f"fn={fn}")
    intersection = tp
    union = tp + fp + fn
    union = np.maximum(union, self.eps)
    # import pdb; pdb.set_trace()
    iou = intersection.astype(np.double) / union.astype(np.double)
    iou_mean = (intersection[self.include].astype(np.double) / union[self.include].astype(np.double)).mean()
    iou_things = (intersection[include_things].astype(np.double) / union[include_things].astype(np.double)).mean()
    if len(include_stuff) == 0:
          iou_stuff = np.array(0.).mean()
    else: iou_stuff = (intersection[include_stuff].astype(np.double) / union[include_stuff].astype(np.double)).mean()
    return iou_mean, iou, iou_things, iou_stuff  # returns "iou mean", "iou per class" ALL CLASSES
  
  def getConfMatrix(self):
    conf = self.px_iou_conf_matrix.copy()
    return conf

  def addBatch(self, x_sem, y_sem):  # x=preds, y=targets
    ''' IMPORTANT: Inputs must be batched. Either [N,H,W], or [N, P]
    '''
    # add to IoU calculation (for checking purposes)
    self.addBatchSemIoU(x_sem, y_sem)

import yaml
DATA = yaml.safe_load(open('configs/label_mapping/semantic-kitti.yaml', 'r'))
# get number of interest classes, and the label mappings
class_remap = DATA["learning_map"]
class_inv_remap = DATA["learning_map_inv"]
class_ignore = DATA["learning_ignore"]
nr_classes = len(class_inv_remap)
class_strings = DATA["labels"]

def printResults(class_evaluator, logger=None):
    class_IoU, class_all_IoU = class_evaluator.getSemIoU()

    # now make a nice dictionary
    output_dict = {}

    # make python variables
    class_IoU = class_IoU.item()
    class_all_IoU = class_all_IoU.flatten().tolist()

    output_dict["all"] = {}
    output_dict["all"]["IoU"] = class_IoU

    for idx, iou in enumerate(class_all_IoU):
        class_str = class_strings[class_inv_remap[idx]]
        output_dict[class_str] = {}
        output_dict[class_str]["IoU"] = iou

    mIoU = output_dict["all"]["IoU"]

    codalab_output = {}

    codalab_output["iou_mean"] = float(mIoU)

    key_list = [
        "iou_mean",
    ]

    if logger != None:
        # evaluated_fnames = class_evaluator.evaluated_fnames
        # logger.info('Evaluated {} frames. Duplicated frame number: {}'.format(len(evaluated_fnames), len(evaluated_fnames) - len(set(evaluated_fnames))))
        logger.info('|        |  IoU   |')
        for k, v in output_dict.items():
            logger.info('|{}| {:.4f} |'.format(
                k.ljust(8)[-8:], v['IoU']))
        return codalab_output
    if logger is None:
        # evaluated_fnames = class_evaluator.evaluated_fnames
        # print('Evaluated {} frames. Duplicated frame number: {}'.format(len(evaluated_fnames), len(evaluated_fnames) - len(set(evaluated_fnames))))
        logger.info('|        |  IoU   |')
        for k, v in output_dict.items():
            logger.info('|{}| {:.4f} |'.format(
                k.ljust(8)[-8:], v['IoU']))
        return codalab_output

    for key in key_list:
        if logger != None:
            logger.info("{}:\t{}".format(key, codalab_output[key]))
        else:
            print("{}:\t{}".format(key, codalab_output[key]))

    return codalab_output

if __name__ == "__main__":
  # generate problem from He paper (https://arxiv.org/pdf/1801.00868.pdf)
  classes = 5  # ignore, grass, sky, person, dog
  cl_strings = ["ignore", "grass", "sky", "person", "dog"]
  ignore = [0]  # only ignore ignore class
  min_points = 1  # for this example we care about all points

  # generate ground truth and prediction
  sem_pred = []
  inst_pred = []
  sem_gt = []
  inst_gt = []

  # some ignore stuff
  N_ignore = 50
  sem_pred.extend([0 for i in range(N_ignore)])
  inst_pred.extend([0 for i in range(N_ignore)])
  sem_gt.extend([0 for i in range(N_ignore)])
  inst_gt.extend([0 for i in range(N_ignore)])

  # grass segment
  N_grass = 50
  N_grass_pred = 40  # rest is sky
  sem_pred.extend([1 for i in range(N_grass_pred)])  # grass
  sem_pred.extend([2 for i in range(N_grass - N_grass_pred)])  # sky
  inst_pred.extend([0 for i in range(N_grass)])
  sem_gt.extend([1 for i in range(N_grass)])  # grass
  inst_gt.extend([0 for i in range(N_grass)])

  # sky segment
  N_sky = 50
  N_sky_pred = 40  # rest is grass
  sem_pred.extend([2 for i in range(N_sky_pred)])  # sky
  sem_pred.extend([1 for i in range(N_sky - N_sky_pred)])  # grass
  inst_pred.extend([0 for i in range(N_sky)])  # first instance
  sem_gt.extend([2 for i in range(N_sky)])  # sky
  inst_gt.extend([0 for i in range(N_sky)])  # first instance

  # wrong dog as person prediction
  N_dog = 50
  N_person = N_dog
  sem_pred.extend([3 for i in range(N_person)])
  inst_pred.extend([35 for i in range(N_person)])
  sem_gt.extend([4 for i in range(N_dog)])
  inst_gt.extend([22 for i in range(N_dog)])

  # two persons in prediction, but three in gt
  N_person = 50
  sem_pred.extend([3 for i in range(6 * N_person)])
  inst_pred.extend([8 for i in range(4 * N_person)])
  inst_pred.extend([95 for i in range(2 * N_person)])
  sem_gt.extend([3 for i in range(6 * N_person)])
  inst_gt.extend([33 for i in range(3 * N_person)])
  inst_gt.extend([42 for i in range(N_person)])
  inst_gt.extend([11 for i in range(2 * N_person)])

  # gt and pred to numpy
  sem_pred = np.array(sem_pred, dtype=np.int64).reshape(1, -1)
  inst_pred = np.array(inst_pred, dtype=np.int64).reshape(1, -1)
  sem_gt = np.array(sem_gt, dtype=np.int64).reshape(1, -1)
  inst_gt = np.array(inst_gt, dtype=np.int64).reshape(1, -1)

  # evaluator
  evaluator = SemanticEval(classes, ignore=ignore, min_points=1)
  evaluator.addBatch(sem_pred, inst_pred, sem_gt, inst_gt)
#   pq, sq, rq, all_pq, all_sq, all_rq = evaluator.getPQ()
  iou, all_iou = evaluator.getSemIoU()

  # [PANOPTIC EVAL] IGNORE:  [0]
  # [PANOPTIC EVAL] INCLUDE:  [1 2 3 4]
  # TOTALS
  # PQ: 0.47916666666666663
  # SQ: 0.5520833333333333
  # RQ: 0.6666666666666666
  # IoU: 0.5476190476190476
  # Class ignore 	 PQ: 0.0 SQ: 0.0 RQ: 0.0 IoU: 0.0
  # Class grass 	 PQ: 0.6666666666666666 SQ: 0.6666666666666666 RQ: 1.0 IoU: 0.6666666666666666
  # Class sky 	 PQ: 0.6666666666666666 SQ: 0.6666666666666666 RQ: 1.0 IoU: 0.6666666666666666
  # Class person 	 PQ: 0.5833333333333333 SQ: 0.875 RQ: 0.6666666666666666 IoU: 0.8571428571428571
  # Class dog 	 PQ: 0.0 SQ: 0.0 RQ: 0.0 IoU: 0.0

  print("TOTALS")
#   print("PQ:", pq.item(), pq.item() == 0.47916666666666663)
#   print("SQ:", sq.item(), sq.item() == 0.5520833333333333)
#   print("RQ:", rq.item(), rq.item() == 0.6666666666666666)
  print("IoU:", iou.item(), iou.item() == 0.5476190476190476)
  for i, iou in enumerate(all_iou):
    print("Class", cl_strings[i], "\t", "IoU:", iou.item())