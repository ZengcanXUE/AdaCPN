import torch
import logging
import numpy as np
from tqdm import tqdm


def eval_for_tail(eval_data, model, device, data, descending, raoit=0):
    hits = []
    hits_left = []
    hits_right = []
    hits_all = []
    ranks = []
    ranks_left = []
    ranks_right = []
    ent_rel_multi_t = data['entity_relation']['as_tail']
    ent_rel_multi_h = data['entity_relation']['as_head']
    for _ in range(10):  # need at most Hits@10
        hits.append([])
        hits_left.append([])
        hits_right.append([])
        hits_all.append([])

    for batch_idx, batch_data in enumerate(tqdm(eval_data)):
        eval_h = batch_data[0].to(device)
        eval_t = batch_data[1].to(device)
        eval_r = batch_data[2].to(device)
        _, pred = model(eval_h, eval_r)  # evaluate corruptions by replacing the object, i.e. tail entity
        _, pred1 = model(eval_t, eval_r, inverse=True)

        # need to filter out the entities ranking above the target entity that form a
        # true (head, tail) entity pair in train/valid/test data
        for i in range(eval_h.size(0)):
            # get all tail entities that form triples with eval_h[i] as the head entity and eval_r[i] as the relation
            filter_t = ent_rel_multi_t[eval_h[i].item()][eval_r[i].item()]
            filter_h = ent_rel_multi_h[eval_t[i].item()][eval_r[i].item()]

            pred_value = pred[i][eval_t[i].item()].item()
            pred_value1 = pred1[i][eval_h[i].item()].item()
            pred[i][filter_t] = 0.0
            pred1[i][filter_h] = 0.0
            pred[i][eval_t[i].item()] = pred_value
            pred1[i][eval_h[i].item()] = pred_value1

        _, index = torch.sort(pred, 1, descending=True)  # pred: (batch_size, ent_count)
        _, index1 = torch.sort(pred1, 1, descending=True)
        index = index.cpu().numpy()  # index: (batch_size)
        index1 = index1.cpu().numpy()

        for i in range(eval_h.size(0)):
            # find the rank of the target entities
            rank = np.where(index[i] == eval_t[i].item())[0][0]
            rank1 = np.where(index1[i] == eval_h[i].item())[0][0]

            # rank+1, since the rank starts with 1 not 0
            ranks_left.append(rank1 + 1)
            ranks_right.append(rank + 1)
            ranks.append(rank1 + 1)
            ranks.append(rank + 1)

            for hits_level in range(10):
                if rank <= hits_level:
                    hits[hits_level].append(1.0)
                else:
                    hits[hits_level].append(0.0)

            for hits_level in range(10):
                if rank1 <= hits_level:
                    hits_left[hits_level].append(1.0)
                else:
                    hits_left[hits_level].append(0.0)


            for hits_level in range(10):
                if rank1 <= hits_level:
                    hits_all[hits_level].append(1.0)
                else:
                    hits_all[hits_level].append(0.0)

            for hits_level in range(10):
                if rank <= hits_level:
                    hits_all[hits_level].append(1.0)
                else:
                    hits_all[hits_level].append(0.0)
            

    return hits, hits_left, ranks, ranks_left, ranks_right, hits_all

def output_eval_tail(epoch, results, data_name):
    hits = np.array(results[0]) 
    hits_left = np.array(results[1]) 
    ranks = np.array(results[2]) 
    ranks_left = np.array(results[3]) 
    ranks_right = np.array(results[4])
    hits_all = np.array(results[5])
    r_ranks_left = 1.0 / ranks_left
    r_ranks_right = 1.0 / ranks_right
    r_ranks = 1.0 / ranks

    tail_hit10 = hits[9].mean()
    tail_hit3 = hits[2].mean()
    tail_hit1 = hits[0].mean()
    tail_MRR = r_ranks_right.mean()
    tail_MR = ranks_right.mean()

    head_hit10 = hits_left[9].mean()
    head_hit3 = hits_left[2].mean()
    head_hit1 = hits_left[0].mean()
    head_MRR = r_ranks_left.mean()
    head_MR = ranks_left.mean()

    all_hit10 = (hits[9].mean() + hits_left[9].mean())/2
    all_hit3 = (hits[2].mean() + hits_left[2].mean())/2
    all_hit1 = (hits[0].mean() + hits_left[0].mean())/2
    all_MRR = (r_ranks_right.mean() + r_ranks_left.mean())/2
    all_MR = (ranks_right.mean() + ranks_left.mean())/2

    print('Tail: For %s data: Hits@10=%.4f - Hits@3=%.4f - Hits@1=%.4f' % (data_name, hits[9].mean(), hits[2].mean(), hits[0].mean()))
    print('Tail: For %s data: MR=%.4f - MRR=%.4f' % (data_name, ranks_right.mean(), r_ranks_right.mean()))

    print('Head: For %s data: Hits@10=%.4f - Hits@3=%.4f - Hits@1=%.4f' % (data_name, hits_left[9].mean(), hits_left[2].mean(), hits_left[0].mean()))
    print('Head: For %s data: MR=%.4f - MRR=%.4f' % (data_name, ranks_left.mean(), r_ranks_left.mean()))
 
    print('All: For %s data: Hits@10=%.4f - Hits@3=%.4f - Hits@1=%.4f' % (data_name, all_hit10, all_hit3, all_hit1))
    print('All: For %s data: MR=%.4f - MRR=%.4f' % (data_name, all_MR, all_MRR))


    if data_name=='test':
        print('[epcoh # %s] %.4f  %.4f  %.4f  %.4f   '
        ' %.4f  %.4f  %.4f  %.4f   '
        ' %.4f  %.4f  %.4f  %.4f   ' % (epoch+1,
                                        r_ranks_right.mean(), hits[9].mean(), hits[2].mean(), hits[0].mean(), 
                                        r_ranks_left.mean(), hits_left[9].mean(), hits_left[2].mean(), hits_left[0].mean(),
                                        all_MRR,all_hit10, all_hit3, all_hit1))
        logging.info('%s\t%.4f\t%.4f\t%.4f\t%.4f\t'
        ' %.4f\t%.4f\t%.4f\t%.4f\t'
        ' %.4f\t%.4f\t%.4f\t%.4f' % (epoch+1,
                                            r_ranks_right.mean(), hits[9].mean(), hits[2].mean(), hits[0].mean(), 
                                            r_ranks_left.mean(), hits_left[9].mean(), hits_left[2].mean(), hits_left[0].mean(),
                                            all_MRR,all_hit10, all_hit3, all_hit1))
    
    
    arr = [tail_hit10, tail_hit3, tail_hit1, tail_MRR, tail_MR, 
           head_hit10, head_hit3, head_hit1, head_MRR, head_MR, 
           all_hit10 , all_hit3 , all_hit1 , all_MRR, all_MR]
    

    return arr
    