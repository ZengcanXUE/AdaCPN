import torch
import time
import argparse
from torch.utils.data import DataLoader
from train import *
from evaluation import *
from models import *
from utils import *
from dataset import *
import datetime
import logging
import os

eval_dict = {
    'eval_for_tail': eval_for_tail
}

class Experiment:
    def __init__(self, config):
        self.model_name = config.get('model_name')
        self.train_conf = config.get('train')
        self.eval_conf = config.get('eval')
        self.dataset = Dataset(config.get('dataset'))
        config['entity_cnt'] = len(self.dataset.data['entity'])
        config['relation_cnt'] = len(self.dataset.data['relation'])
        config['data'] = self.dataset.data['train']
        self.model, self.device = init_model(config)
        self.eval_func = eval_dict[self.eval_conf.get('eval_func')]
        if self.model_name in ['AdaCPN']:
            self.train_func = train_without_label
            self.output_func = output_eval_tail

        else:
            logging.error(f'Could not find any training function for model={self.model_name}')
        opt_conf = config.get('optimizer')
        if opt_conf.get('algorithm') == 'adam':
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=opt_conf.get('lr'), weight_decay=opt_conf.get('weight_decay'))
        elif opt_conf.get('algorithm') == 'sgd':
            self.optimizer = torch.optim.SGD(self.model.parameters(), lr=opt_conf.get('lr'), weight_decay=opt_conf.get('weight_decay'))
        elif opt_conf.get('algorithm') == 'adagrad':
            self.optimizer = torch.optim.Adagrad(self.model.parameters(), lr=opt_conf.get('lr'), weight_decay=opt_conf.get('weight_decay'))
        else:
            logging.error('Could not find corresponding optimizer for algorithm={}'.format(opt_conf.get('algorithm')))
        self.save_model_path = config.get('save_model_path')
    
    def train_and_eval(self):
        train_loader = DataLoader(self.dataset.data['train'], self.train_conf.get('batch_size'), shuffle=self.train_conf.get("shuffle"), drop_last=False)
        if self.dataset.data['valid']:
            valid_loader = DataLoader(self.dataset.data['valid'], self.eval_conf.get('batch_size'), shuffle=False, drop_last=False)
        if self.dataset.data['test']:
            test_loader = DataLoader(self.dataset.data['test'], self.eval_conf.get('batch_size'), shuffle=False, drop_last=False)
        last_tail_score = -1.0
        best_tail_H10 = 0
        best_tail_H3 = 0
        best_tail_H1 = 0
        best_tail_MRR = 0
        best_tail_MR = 0

        last_head_score = -1.0
        best_head_H10 = 0
        best_head_H3 = 0
        best_head_H1 = 0
        best_head_MRR = 0
        best_head_MR = 0

        last_all_score = -1.0
        best_all_H10 = 0
        best_all_H3 = 0
        best_all_H1 = 0
        best_all_MRR = 0
        best_all_MR = 0
        print('Epoch\t[Tail]MRR\tHits@10\tHits@3\tHits@1\t[Head]MRR\tHits@10\tHits@3\tHits@1\t[All]MRR\tHits@10\tHits@3\tHits@1')
        logging.info('Epoch\t[Tail]MRR\tHits@10\tHits@3\tHits@1\t[Head]MRR\tHits@10\tHits@3\tHits@1\t[All]MRR\tHits@10\tHits@3\tHits@1')
    
        for epoch in range(self.train_conf.get('epochs')):
            start_time = time.time()
            epoch_loss = self.train_func(train_loader, self.model, self.optimizer, self.device)
            end_time = time.time()
            mean_loss = np.mean(epoch_loss)
            print('[Epoch #%d] training loss: %f - training time: %.2f seconds' % (epoch + 1, mean_loss, end_time - start_time))
            
            
            if self.eval_conf.get('do_validate') and (epoch + 1) % self.eval_conf.get('valid_steps') == 0:
                print(f'--- epoch #{epoch + 1} valid ---')
                self.model.eval()
                with torch.no_grad():
                    eval_results = self.eval_func(valid_loader, self.model, self.device, self.dataset.data, self.eval_conf.get('scoring_desc'))
                    self.output_func(epoch, eval_results, 'validation')

                if epoch > -1 :
                    print(f'--- test ---')
                    self.model.eval()
                    with torch.no_grad():
                        eval_results = self.eval_func(test_loader, self.model, self.device, self.dataset.data, self.eval_conf.get('scoring_desc'))
                        arr = self.output_func(epoch, eval_results, 'test')
                        tail_arr = arr[0:5]
                        head_arr = arr[5:10] 
                        all_arr = arr[10:15]  
                        
                        now_tail_score = sum(tail_arr[0:4])
                        now_head_score = sum(head_arr[0:4])
                        now_all_score = sum(all_arr[0:4])
                    if now_tail_score > last_tail_score :
                        last_tail_score = now_tail_score
                        best_tail_H10 = tail_arr[0]
                        best_tail_H3 = tail_arr[1]
                        best_tail_H1 = tail_arr[2]
                        best_tail_MRR = tail_arr[3]
                        best_tail_MR = tail_arr[4]
                        if not os.path.exists(self.save_model_path):
                            os.makedirs(self.save_model_path)
                            logging.info('Created output directory {}'.format(self.save_model_path))
                        torch.save(self.model, f'{self.save_model_path}/{self.model_name}_{self.dataset.name}.ckpt')

                    if now_head_score > last_head_score :
                        last_head_score = now_head_score
                        best_head_H10 = head_arr[0]
                        best_head_H3 = head_arr[1]
                        best_head_H1 = head_arr[2]
                        best_head_MRR = head_arr[3]
                        best_head_MR = head_arr[4]

                    if now_all_score > last_all_score :
                        last_all_score = now_all_score
                        best_all_H10 = all_arr[0]
                        best_all_H3 = all_arr[1]
                        best_all_H1 = all_arr[2]
                        best_all_MRR = all_arr[3]
                        best_all_MR = all_arr[4]
        
        
        print(f'Best tail test score: MR={best_tail_MR:.4f} - MRR={best_tail_MRR:.4f} - Hits@10={best_tail_H10:.4f} - Hits@3={best_tail_H3:.4f} - Hits@1={best_tail_H1:.4f}')
        print(f'Best head test score: MR={best_head_MR:.4f} - MRR={best_head_MRR:.4f} - Hits@10={best_head_H10:.4f} - Hits@3={best_head_H3:.4f} - Hits@1={best_head_H1:.4f}')
        print(f'Best all test score: MR={best_all_MR:.4f} - MRR={best_all_MRR:.4f} - Hits@10={best_all_H10:.4f} - Hits@3={best_all_H3:.4f} - Hits@1={best_all_H1:.4f}')
        print('Finished!')

        logging.info('Best tail test score: MR=%.4f - MRR=%.4f - Hits@10=%.4f - Hits@3=%.4f - Hits@1=%.4f',best_tail_MR,best_tail_MRR, best_tail_H10, best_tail_H3, best_tail_H1)
        logging.info('Best head test score: MR=%.4f - MRR=%.4f - Hits@10=%.4f - Hits@3=%.4f - Hits@1=%.4f',best_head_MR,best_head_MRR, best_head_H10, best_head_H3, best_head_H1)
        logging.info('Best all test score: MR=%.4f - MRR=%.4f - Hits@10=%.4f - Hits@3=%.4f - Hits@1=%.4f',best_all_MR,best_all_MRR, best_all_H10, best_all_H3, best_all_H1)
        logging.info('Finished!')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Knowledge graph inference arguments.')
    parser.add_argument('-c', '--config', dest='config_file', help='The path of configuration json file.')
    args = parser.parse_args()
    print(args)

    
    
    config = load_json_config(args.config_file)
    print(config)
    logging.info(args)
    logging.info(config)
    
    
    experiment = Experiment(config)

    experiment.train_and_eval()

