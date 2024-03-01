

import os,pickle
# os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import numpy as np
import torch
from torch import optim
import torch.nn as nn
from models import UNet
from tqdm import tqdm
from utils.tools import read_config, check_dir,seed_everything
from collections import OrderedDict
from models import get_model
from datasets import get_loaders
from losses import *
from utils import *

torch.backends.cudnn.benchmark = True
torch.backends.cudnn.enabled = True
seed_everything(666)
DEVICE=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# DEVICE=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

#总变量列表


class set_model(object):  
    def __init__(self,config,mode='train',task=None,test_names=None,debug=False,preload=False):
        self.debug=debug
        self.model_name=None

        config=os.path.join(os.path.join(os.path.dirname(__file__)),'configs',config)
        self.configs = read_config(config)
        # self.model=UPerNet(len(NAMES),1)
        self.task=task

        self.model=get_model(self.configs)
        data_path=self.configs.data_path
        if mode =='test':
            self.test_loader=get_loaders()

        if mode in ['train','valid']:
            if mode=='train':
                self.train_loader=get_loaders()
            self.valid_loader=get_loaders()
            

            self.optimizer = optim.AdamW(self.model.parameters(), lr=self.configs.learning_rate)
            if hasattr(self.configs, self.configs.scheduler):
                params = getattr(self.configs, self.configs.scheduler)
            else:
                raise ValueError(f'No support {self.configs.scheduler} scheduler!')
            self.lr_sched=LRScheduler(self.optimizer, **params)
            self.earlystop=EarlyStopping(patience=10, min_delta=0)
            self.criterion=get_loss()
        
        self.preload=preload
        #验证或预报模式，则载入pth
        if mode in ['valid','test'] or preload: 
            scalars_save_dir=os.path.join(self.configs.result_path,'scalars')
            check_dir(scalars_save_dir)
            model_save_dir=os.path.join(self.configs.result_path,'pth')
            check_dir(model_save_dir)
            with open(os.path.join(scalars_save_dir,"scalars.pkl"), 'rb') as f:
                dicts=pickle.load(f)
            bestepoch=dicts['best_epoch'] #获取最佳epoch序号
            # bestepoch=5
            print('bestopoch:',bestepoch,'train_loss:',dicts['train_loss'][bestepoch],'valid_loss:',dicts['valid_loss'][bestepoch],'score:',dicts['ls_score'][bestepoch])
            model_path=os.path.join(model_save_dir,f'epoch_{bestepoch}.pth')
            self.load_model(model_path)
        else:
            if torch.cuda.device_count() > 1:
                    self.model = nn.DataParallel(self.model)
            self.model=self.model.to(DEVICE) 

    def load_model(self,pth_path=None):
        load_model=torch.load(pth_path,map_location=DEVICE)
        # self.model.load_state_dict(load_model)
        
        if torch.cuda.device_count() > 1:
                print(torch.cuda.device_count() )
                new_state_dict=OrderedDict()
                for k, v in load_model.items(): # k为module.xxx.weight, v为权重
                    name = k[7:] # 截取`module.`后面的xxx.weight
                    new_state_dict[name] = v
                self.model.load_state_dict(new_state_dict)
                self.model = nn.DataParallel(self.model)
        else:
            self.model.load_state_dict(load_model)
        self.model=self.model.to(DEVICE) 

    def preload_param(self,):
        scalars_save_dir=os.path.join(self.configs.result_path,'scalars')
        with open(os.path.join(scalars_save_dir,"scalars.pkl"), 'rb') as f:
            dicts=pickle.load(f)
        ls_train_loss=list(dicts['train_loss'])
        ls_val_loss=list(dicts['valid_loss'])
        ls_score=list(dicts['ls_score'])
        bestepoch=dicts['best_epoch']
        epochnow=len(ls_score)
        return ls_train_loss,ls_val_loss,ls_score,bestepoch,epochnow

    def run(self):     
        check_dir(self.configs.result_path)
        result_dir=self.configs.result_path
        model_save_dir=os.path.join(result_dir,'pth')
        scalars_save_dir=os.path.join(result_dir,'scalars') 
        check_dir(model_save_dir)
        check_dir(scalars_save_dir)   

        ls_score=[]
        ls_val_loss=[]
        ls_train_loss=[]
        train_loss = 0.0
        bestepoch=0
        best_score=9999.  
        epochnow=0  
        if self.preload:
            ls_train_loss,ls_val_loss,ls_score,bestepoch,epochnow=self.preload_param()
            best_score=ls_score[bestepoch]['score']
            best_score_all=ls_score[bestepoch]

        for epoch in range(epochnow,self.configs.num_epochs):    #cfg.NUM_EPOCHS
            print(f"{self.configs.task}=> epoch:",epoch)
            train_loss=self.train_fn()
            print("train_loss:",train_loss)  
            valid_loss,scores=self.validate_fn()
            ls_score.append(scores)
            self.lr_sched(scores['score'])        
            self.earlystop(scores['score'])
            if self.earlystop.early_stop:
                break
            if best_score>scores['score']:
                best_score=scores['score']
                best_score_all=scores
                bestepoch=epoch
                #保存当前epoch模型       
                if model_save_dir is not None:
                    if not os.path.exists(model_save_dir):
                        os.mkdir(model_save_dir)
                    # else:
                    #     shutil.rmtree(model_save_dir)
                    #     os.mkdir(model_save_dir)
                torch.save(self.model.state_dict(), os.path.join(model_save_dir, f'epoch_{epoch}.pth'))    
            ls_val_loss.append(valid_loss)   
            ls_train_loss.append(train_loss)

            #保存各类参数
            if scalars_save_dir is not None:
                dicts={"train_loss":np.array(ls_train_loss),"valid_loss":np.array(ls_val_loss),
                        "ls_score":ls_score,'best_epoch':bestepoch,'best_score':best_score_all}
                with open(os.path.join(scalars_save_dir,"scalars.pkl"), 'wb') as f:
                    pickle.dump(dicts, f, pickle.HIGHEST_PROTOCOL) #保存文件

    def train_fn(self,):
        train_loss = 0.0
        self.model.train()
        count=0
        scaler = torch.cuda.amp.GradScaler()#加快运算
        loop = tqdm(self.train_loader)
        for batch_idx, (inputs, targets) in enumerate(loop):      
            t=torch.tensor(range(1,self.configs.target_length+1,1))
            t=t[None,:].repeat(targets.shape[0],1)
            t=t.view(-1).to(DEVICE)
            inputs=inputs.to(DEVICE)
            targets=targets.to(DEVICE)  
            B,T,C,H,W=inputs.shape
            inputs=inputs.view(B,T*C,H,W)
            inputs=inputs.unsqueeze(1).repeat(1,self.configs.target_length,1,1,1)
            inputs=inputs.view(B*self.configs.target_length,T*C,H,W)
            # time=time.to(DEVICE).unsqueeze(1).repeat(1,self.configs.target_length,1)
            # time=time.view(B*self.configs.target_length,-1)
            with torch.cuda.amp.autocast():
                output = self.model(inputs,t)
                output=output.view(B,self.configs.target_length,self.out_chans,H,W)
                mask=torch.zeros_like(targets)
                mask[:,:,:,30:-30,30:-30]=1.
                loss=self.mae(output,targets,mask)
                # output=output.squeeze()
                # mask=torch.ones_like(targets)
                # mask[targets<-9999]=0.  
                # loss=self.c_hour(output,targets,mask)
                self.optimizer.zero_grad()
                scaler.scale(loss).backward()
                scaler.step(self.optimizer)
                scaler.update()

            loop.set_postfix(loss=loss.item())
            train_loss+=loss.item()
            count+=1
            if self.debug:
                if count>2:break
        train_loss=train_loss/count
        return train_loss

    
    
    def validate_fn(self,):
        with torch.no_grad():
            self.model.eval()
            valid_loss = 0.0
            valid_time = 0
            
        return valid_loss
    


def test():
    data_dir='/workspace/AI_datas/RamData/Datas/'
    loader=get_loaders(data_dir,batch_size=4,num_works=4)
    # DEVICE=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model=UNet(img_channels=70*2,
        base_channels=64,output_channels=5)
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
    model=model.to(DEVICE) 
    for input, target,time in tqdm(loader):
        t=torch.tensor(range(1,21,1))
        t=t[None,:].repeat(time.shape[0],1)
        t=t.view(-1).to(DEVICE)
        input=input.to(DEVICE)
        target=target.to(DEVICE)  
        B,T,C,H,W=input.shape
        input=input.view(B,T*C,H,W)
        input=input.unsqueeze(1).repeat(1,20,1,1,1)
        input=input.view(B*20,T*C,H,W)
        time=time.to(DEVICE).unsqueeze(1).repeat(1,20,1)
        time=time.view(B*20,-1)
        output = model(input,time,t)
        # output=output.view(B,20,5,H,W)

if __name__=='__main__':
    mymodel=set_model(mode='train',task=['tp'],debug=False,preload=False)
    mymodel.run()
    # test()
  