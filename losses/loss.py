
from torch import nn
import torch
import torch.nn.functional as F
import numpy as np

class Weighted_mse_mae(nn.Module):
    def __init__(self, mse_weight=1.0, mae_weight=1.0,thresholds=[30,40,50],balancing_weights=[1,5,10,30], NORMAL_LOSS_GLOBAL_SCALE=0.005):
        super().__init__()
        self.NORMAL_LOSS_GLOBAL_SCALE = NORMAL_LOSS_GLOBAL_SCALE
        # MSE,MAE 分别给与权重
        self.mse_weight = mse_weight
        self.mae_weight = mae_weight
        # 每一帧 Loss 递进参数
        self._thresholds=thresholds
        self._weight=balancing_weights

    # __C.RAIN.THRESHOLDS = [0.1, 3, 10, 20]
    # __C.BALANCING_WEIGHTS = ( 1, 2, 3, 4, 5)
    def forward(self, input, target, mask=None):
        #cfg.HKO.EVALUATION.BALANCING_WEIGHTS=(1, 1, 2, 5, 10, 30,60)
        #balancing_weights = cfg.HKO.EVALUATION.BALANCING_WEIGHTS 
        balancing_weights = self._weight
        weights = torch.ones_like(input) * balancing_weights[0]
        thresholds=self._thresholds
        #set weight to every pixel
        for i, threshold in enumerate(thresholds):
            weights = weights + (balancing_weights[i + 1] - balancing_weights[i]) * (target >= threshold).float()
        #what is mask??mask 是一个0、1掩码矩阵，与weights点积后，mask为0部分weight为0
        if mask is not None:
            weights = weights * mask.float()
        # input: S*B*1*H*W
        # error: S*B
        mse = torch.mean(weights * ((input-target)**2))
        mae = torch.mean(weights * (torch.abs((input-target))))
        return self.NORMAL_LOSS_GLOBAL_SCALE * (self.mse_weight*mse+ self.mae_weight*mae)

class Weighted_mse(nn.Module):
    def __init__(self, thresholds=[30,40,50],balancing_weights=[1,5,10,30], NORMAL_LOSS_GLOBAL_SCALE=1):
        super().__init__()
        self.NORMAL_LOSS_GLOBAL_SCALE = NORMAL_LOSS_GLOBAL_SCALE
        # 每一帧 Loss 递进参数
        self._thresholds=thresholds
        self._weight=balancing_weights

    def forward(self, input, target, mask=None):
        balancing_weights = self._weight
        weights = torch.ones_like(input) * balancing_weights[0]
        thresholds=self._thresholds
        #set weight to every pixel
        for i, threshold in enumerate(thresholds):
            weights = weights + (balancing_weights[i + 1] - balancing_weights[i]) * (target >= threshold).float()
        #what is mask??mask 是一个0、1掩码矩阵，与weights点积后，mask为0部分weight为0
        if mask is not None:
            weights = weights * mask.float()
        mse = torch.mean(weights * ((input-target)**2))
        return self.NORMAL_LOSS_GLOBAL_SCALE *mse

class class_mse(nn.Module):
    def __init__(self, thresholds=[30,40,50],balancing_weights=None, NORMAL_LOSS_GLOBAL_SCALE=1):
        super().__init__()
        self.NORMAL_LOSS_GLOBAL_SCALE = NORMAL_LOSS_GLOBAL_SCALE
        # 每一帧 Loss 递进参数
        self._thresholds=thresholds
        self._weight=balancing_weights
    
    def class_mse(self,inputs,target,mask,weight):
        mse=(weight*mask*(inputs-target)**2).sum()/(torch.sum(mask)+0.00000001)
        return mse
    
    def class_mae(self,inputs,target,mask,weight):
        mae=(weight*mask*torch.abs(inputs-target)).sum()/(torch.sum(mask)+0.00000001)
        return mae

    def transfer_to_OR(self,target):
        target=target.unsqueeze(1)
        OR_layers=[]
        for thre in self._thresholds:
            OR_layers.append((target>=thre).float())
        OR_layers=torch.cat(OR_layers,dim=1)
        return OR_layers

    def forward(self, input, target, mask=None):
        weights = self._weight

        # ls=[]
        # for thre in self._thresholds:
        #     ls.append(input.unsqueeze(1)-thre)
        # inputs_sig=torch.cat(ls,dim=1)
        # inputs_sig=torch.sigmoid(inputs_sig)
        thresholds=self._thresholds
        class_index=self.transfer_to_OR(target)
        #set weight to every pixel
        if weights is None:
            weights = [1] * len(self._thresholds)
        
        loss=0.0
        for i in range( len(self._thresholds)):
            # loss += dice * weight[i]
            maskin=torch.ones_like(input)
            maskin[input<self._thresholds[i]]=0.
            maskmerge=maskin+class_index[:, i]
            maskmerge=torch.where(maskmerge>=1,1.,0.)
            if mask is not None:
                weight = weights[i] * mask.float()
            mae=self.class_mae(input,target,maskmerge,weight)
            # mse=mse/(1+self._thresholds[i])
            
            loss += (mae*weights[i])
        # input: S*B*1*H*W
        # error: S*B
        return loss/(len(thresholds))*0.01

class Masked_mae(nn.Module):
    def __init__(self,):
        super().__init__()

    def forward(self, input, target, mask):
        mae=torch.sum(torch.abs(input-target)*mask)/mask.sum()
        mse=torch.sum(mask*(input-target)**2)/mask.sum()    
        return (mae+mse)/2
    
class Masked_SME(nn.Module):
    def __init__(self,):
        super().__init__()

    def forward(self, input, target, mask):
        sme=torch.sum((torch.abs(input-target)+1e-8)/(torch.abs(input+target)+1e-8)*mask)/mask.sum()  
        return sme

class ORDiceLoss_diff(nn.Module):
    def __init__(self,thresholds, weight=None,cosh_log=False,add_loss=None):
        super(ORDiceLoss_diff, self).__init__()
        self._thresholds=list(np.array(thresholds))
        self._weights=weight
        self.cosh_log=cosh_log
        self.losstype=add_loss
        if add_loss is not None:
            assert add_loss in ['ts','hss','ts+hss','hss+ts']
        

    def transfer_to_OR(self,target):
        target=target.unsqueeze(1)
        # print(target.max())
        OR_layers=[]
        for thre in self._thresholds:
            OR_layers.append((target>=thre).float())
        OR_layers=torch.cat(OR_layers,dim=1)
        # print(OR_layers.sum())
        return OR_layers.float()

    def _dice_loss(self, score, target,mask):
        # target = target.float()
        smooth = 1e-8
        intersect = torch.sum(score * target*mask)  
        y_sum = torch.sum(target *mask)
        z_sum = torch.sum(score *mask)
        # if intersect==0:
        #     print('score:',z_sum,'target',y_sum,'mask:',mask.sum().item())
        loss = (2 * intersect + smooth) / (z_sum + y_sum + smooth)
        loss = 1 - loss
        return loss

    def _cosh_log(self,dice_loss):
        return torch.log(torch.cosh(dice_loss))
    
    def class_mse(self,inputs,target,mask):
        mse=(mask*(inputs-target)**2).sum()/(torch.sum(mask)+0.00000001)
        return mse

    def ts_loss(self,preds,targets,threshold):
        gt=targets.float()
        ls=(1-targets).float()
        sigh=torch.sigmoid(preds-threshold)
        sigm=torch.sigmoid(threshold-preds)
        NA=(gt*sigh).sum() # hits
        NC=(gt*sigm).sum() # misses
        NB=(ls*sigh).sum() # false alarm
        ND=(ls*sigm).sum() # true negative
        N=targets.numel()
        smooth=1e-8
        if self.losstype=='ts':
            ts=(NA+smooth)/(NA+NB+NC+smooth)
            lo=1-ts
        elif self.losstype=='hss':
            expect=((NA+NC)*(NA+NB)+(ND+NC)*(ND+NB))/N
            hss=((NA+ND)-expect+smooth)/(N-expect+smooth)
            lo=1-hss
        elif self.losstype in ['ts+hss','hss+ts']:
            ts=(NA+smooth)/(NA+NB+NC+smooth)
            expect=1/N*((NA+NC)*(NA+NB)+(ND+NC)*(ND+NB))
            hss=((NA+ND)-expect+smooth)/(N-expect+smooth)
            lo=1-ts+(1-hss)/2
        return lo

    def forward(self, inputs, target,mask):
        class_index=self.transfer_to_OR(target)
        loss=0.0
        for i in range(0, len(self._thresholds)):
            input_sig=torch.sigmoid(inputs-self._thresholds[i])
            dice = self._dice_loss(input_sig, class_index[:, i],mask)
            if self.cosh_log:
                dice=self._cosh_log(dice)
            loss += dice
        return loss / len(self._thresholds)

class ORDiceLoss(nn.Module):
    def __init__(self,thresholds, weight=None,cosh_log=False):
        super(ORDiceLoss, self).__init__()
        self._thresholds=thresholds
        self._weights=weight
        self.cosh_log=cosh_log
        self.n_classes=len(thresholds)

    def transfer_to_OR(self,target):
        target=target.unsqueeze(2)
        OR_layers=[]
        for thre in self._thresholds:
            OR_layers.append((target>=thre).float())
        OR_layers=torch.cat(OR_layers,dim=2)
        return OR_layers

    def _one_hot_encoder(self, input_tensor):
        tensor_list = []
        for i in range(1,len(self._thresholds)+1):
            temp_prob = input_tensor == i  # * torch.ones_like(input_tensor)
            tensor_list.append(temp_prob.unsqueeze(2))
        output_tensor = torch.cat(tensor_list, dim=2)
        return output_tensor.float()

    def _dice_loss(self, score, target,mask):
        target = target.float()
        smooth = 1e-5
        intersect = torch.sum(score * target*mask)
        y_sum = torch.sum(target *mask)
        z_sum = torch.sum(score *mask)
        loss = (2 * intersect + smooth) / (z_sum + y_sum + smooth)
        loss = 1 - loss
        return loss

    def _cosh_log(self,dice):
        return torch.log((torch.exp(dice)+torch.exp(-dice))/2.0)

    def forward(self, inputs, target,mask,  softmax=True):
        # weight=self._weights
        if softmax:
            inputs = torch.sigmoid(inputs)
        target=self.transfer_to_OR(target)

        # else: weight=weight[class_index]
        assert inputs.size() == target.size(), 'predict {} & target {} shape do not match'.format(inputs.size(), target.size())
        loss = 0.0
        for i in range(0, self.n_classes):
            dice = self._dice_loss(inputs[:,:, i], target[:,:, i],mask)
            if self.cosh_log:
                dice=self._cosh_log(dice)
            loss += dice 
        return loss / self.n_classes

class BCELoss(torch.nn.Module):
    def __init__(self, threshold=0.0, weight=2):
        super(BCELoss, self).__init__()
        self.threshold=threshold
        self.elipson = 0.000001
        self.weight=weight

    def forward(self, predict, target,mask):
        threshold=self.threshold
        if threshold==0.0 :labels=torch.where(target==0.0,1,0)
        else:labels=torch.where(target>=threshold,1,0)  
        # mask=torch.ones_like(target)
        # mask[target<-9999]=0
        pt = torch.sigmoid(predict) # sigmoide获取概率

        loss=-(self.weight*(pt+self.elipson).log()*labels+(1-labels)*(1-pt+self.elipson).log())
   
        loss =(loss*mask).sum()/(mask>0).sum()
        return loss

class ORBCELoss(torch.nn.Module):
    def __init__(self, threshold=0.0, weight=None):
        super(ORBCELoss, self).__init__()
        self.threshold=threshold
        self.elipson = 0.000001
        self.weight=weight
    
    def transfer_to_OR(self,target):
        target=target.unsqueeze(1)
        OR_layers=[]
        for thre in self.threshold:
            OR_layers.append((target>=thre).float())
        OR_layers=torch.cat(OR_layers,dim=1)
        return OR_layers
    def forward(self, predict, target):
        mask=torch.ones_like(target)
        mask[target<-9999]=0
        orlayers=self.transfer_to_OR(target)
        B,C,H,W=predict.size()
        pt = torch.sigmoid(predict) # sigmoide获取概率
        loss=torch.zeros_like(target)
        for i in range(C):
            los=-(self.weight[i]*(pt[:,i,...]+self.elipson).log()*orlayers[:,i,...]+(1-orlayers[:,i,...])*(1-pt[:,i,...]+self.elipson).log())
            loss+=los 
        loss =(loss*mask).sum()/(mask>0).sum()*0.1
        return loss

class DiceLoss(nn.Module):
    def __init__(self, n_classes,thresholds,weight=None):
        super(DiceLoss, self).__init__()
        self.n_classes = n_classes
        self._thresholds=thresholds
        self._weights=None

    def _one_hot_encoder(self, input_tensor):
        tensor_list = []
        for i in range(self.n_classes):
            temp_prob = input_tensor == i  # * torch.ones_like(input_tensor)
            tensor_list.append(temp_prob.unsqueeze(1))
        output_tensor = torch.cat(tensor_list, dim=1)
        return output_tensor.float()

    def _dice_loss(self, score, target):
        target = target.float()
        smooth = 1e-5
        intersect = torch.sum(score * target)
        y_sum = torch.sum(target * target)
        z_sum = torch.sum(score * score)
        loss = (2 * intersect + smooth) / (z_sum + y_sum + smooth)
        loss = 1 - loss
        return loss

    def forward(self, inputs, target,  softmax=True):
        weight=self._weights
        if softmax:
            inputs = torch.softmax(inputs, dim=1)
        class_index = torch.zeros_like(target).long()
        thresholds = [0.0] + list(self._thresholds)
        # print(thresholds)
        for i, threshold in enumerate(thresholds):
            class_index[target >= threshold] = i
        target = self._one_hot_encoder(class_index)
        if weight is None:
            weight = [1] * self.n_classes
        # else: weight=weight[class_index]
        assert inputs.size() == target.size(), 'predict {} & target {} shape do not match'.format(inputs.size(), target.size())
        class_wise_dice = []
        loss = 0.0
        for i in range(0, self.n_classes):
            dice = self._dice_loss(inputs[:, i], target[:, i])
            class_wise_dice.append(1.0 - dice.item())
            loss += dice * weight[i]
        return loss / self.n_classes

class Diff_loss(nn.Module):
    '''
        losstype=hss/hss+bias/ts/ts+bias/bias,loss计算类型
        thresholds:等级阈值
        weight:各等级权重
    '''
    def __init__(self, thresholds,weight=None,losstype='hss'):
        super().__init__()
        assert losstype in ['hss','hss+bias','ts','ts+bias','ts+hss','bias'],'losstype setting is wrong'
        self.thresholds = thresholds
        self.weight=weight
        self.losstype=losstype


    def _loss(self,preds,targets,threshold):
        gt=(targets>=threshold).float()
        ls=(targets<threshold).float()
        sigh=torch.sigmoid(preds-threshold)
        sigm=torch.sigmoid(threshold-preds)
        NA=(gt*sigh).sum() # hits
        NC=(gt*sigm).sum() # misses
        NB=(ls*sigh).sum() # false alarm
        ND=(ls*sigm).sum() # true negative
        N=targets.numel()
        smooth=1e-8
        if self.losstype=='ts':
            ts=(NA+smooth)/(NA+NB+NC+smooth)
            lo=1-ts
        elif self.losstype=='hss':
            expect=((NA+NC)*(NA+NB)+(ND+NC)*(ND+NB))/N
            hss=((NA+ND)-expect+smooth)/(N-expect+smooth)
            lo=1-hss
        elif self.losstype=='bias':
            bias=(NA+NB+smooth)/(NA+NC+smooth)
            bias=(torch.tanh(torch.log(bias)))**2
            lo=bias
        elif self.losstype in ['hss+bias','bias+hss']:
            expect=1/N*((NA+NC)*(NA+NB)+(ND+NC)*(ND+NB))
            hss=((NA+ND)-expect+smooth)/(N-expect+smooth)
            bias=(NA+NB+smooth)/(NA+NC+smooth)
            bias=(torch.tanh(torch.log(bias)))**2
            lo=1-hss+bias
        elif self.losstype in ['ts+bias','bias+ts']:
            ts=(NA+smooth)/(NA+NB+NC+smooth)
            bias=(NA+NB+smooth)/(NA+NC+smooth)
            bias=(torch.tanh(torch.log(bias)))**2
            lo=1-ts+bias

        elif self.losstype in ['ts+hss','hss+ts']:
            ts=(NA+smooth)/(NA+NB+NC+smooth)
            expect=1/N*((NA+NC)*(NA+NB)+(ND+NC)*(ND+NB))
            hss=((NA+ND)-expect+smooth)/(N-expect+smooth)
            lo=1-ts+(1-hss)/2
        return lo

    def forward(self,preds,targets):
        lo=0
        if self.weight is None:
            self.weight=[1]*len(self.thresholds)
        for i,thre in enumerate(self.thresholds):
            lo+=(self._loss(preds,targets,thre)*self.weight[i])
        return lo




rain_thre=[0.5,1.,2.,10.]
r2=[5.,10.,20.,100.]
echo_thre=[30.,40.,50.,70.]
wind_thre=[5.,10.8,17.2,35]


class testmodel(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv=nn.Conv2d(20,20,3,1,1)
    def forward(self,x):
        x=self.conv(x)
        return x

if __name__=='__main__':
    model=testmodel()
    criterion=Diff_loss(rain_thre,losstype='ts')
    c1=Diff_loss(r2,losstype='ts')
    c2=Diff_loss(r2,losstype='ts')
    c3=Diff_loss(r2,losstype='ts')
    optimizer=torch.optim.Adam(model.parameters(),lr=1e-3)
    y=torch.randn(16,20,10,10)*10
    for i in range(100000):
        optimizer.zero_grad()
        x=model(y)
        x1=x[:,:10,...].sum(dim=1)
        x2=x[:,5:15,...].sum(dim=1)
        x3=x[:,10:20,...].sum(dim=1)
        loss=criterion(x,y)+c1(x1,y[:,:10,...].sum(dim=1))+\
        c2(x2,y[:,5:15,...].sum(dim=1))+c3(x3,y[:,10:20,...].sum(dim=1))
        loss.backward()
        optimizer.step()
        print(loss.item())
