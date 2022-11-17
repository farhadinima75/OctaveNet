import torch, torch.nn as nn, torch.nn.functional as F, os, glob, shutil, numpy as np, sys, tqdm, cv2, time, gc, random
from torchvision import transforms as T
from torch.nn import init
from torch import Tensor
import torchvision.transforms.functional as TF, random, pytz, datetime
from IPython.display import clear_output

from Modules.Metrics import *
from Modules.Model import OctaveNet

def PrepareLibrariesAndDatasets():
  os.system("pip install torch==1.11.0+cu113 torchvision==0.12.0+cu113 torchaudio==0.11.0 --extra-index-url https://download.pytorch.org/whl/cu113")
  os.system("apt update")
  os.system("apt install aria2")
  DatasetPath = '/content/Datasets'
  ModelPath   = '/content/Models'
  os.makedirs(DatasetPath, exist_ok=True)
  os.system("aria2c -x 8 -d /content/Datasets -o LEVIR-CD-256.zip    https://emailkntuacir-my.sharepoint.com/:u:/g/personal/farhadinima75_email_kntu_ac_ir/ERDI-5yFHahDl-1a390i2DgBJULY_BNh6vctHm4swinNZg?download=1")
  os.system("aria2c -x 8 -d /content/Datasets -o DSIFN-CD-256.zip    https://emailkntuacir-my.sharepoint.com/:u:/g/personal/farhadinima75_email_kntu_ac_ir/EeMlQQld78pGgCq3puurff0B8yrzo0qkN1-q0DfdpLDCFw?download=1")
  os.system("aria2c -x 8 -d /content/Datasets -o WHU-CD-256.zip      https://emailkntuacir-my.sharepoint.com/:u:/g/personal/farhadinima75_email_kntu_ac_ir/Ebh6rPs8y1xAi4In2D8Crc4BvRIy7_kSUj7EvNRfPcAKyQ?download=1")
  os.system("aria2c -x 8 -d /content/Datasets -o CropLand-CD-256.zip https://emailkntuacir-my.sharepoint.com/:u:/g/personal/farhadinima75_email_kntu_ac_ir/EVUpqzbgHF9Clr8pdNdaorQBWJQnDaojrHkcNDTBmRIojw?download=1")

  os.system("aria2c -x 8 -d /content/Models -o OctaveNet8_WHU.pt  https://emailkntuacir-my.sharepoint.com/:u:/g/personal/farhadinima75_email_kntu_ac_ir/EYZL5xnBYutBvbmw5LITkF8BPcjFWxvPpz7f6XdG2ER-zg?download=1")
  os.system("aria2c -x 8 -d /content/Models -o OctaveNet32_WHU.pt https://emailkntuacir-my.sharepoint.com/:u:/g/personal/farhadinima75_email_kntu_ac_ir/EQ55tvZXi49DmOlEIwezxVcB9LoMQSevNIQvbaMs3z4DYg?download=1")
  os.system("aria2c -x 8 -d /content/Models -o OctaveNet64_WHU.pt https://emailkntuacir-my.sharepoint.com/:u:/g/personal/farhadinima75_email_kntu_ac_ir/Ec0Ait83aZtFvMKdHJ1LTqIBViukYf5L5ppqrAgTfgz1zQ?download=1")
  os.system("aria2c -x 8 -d /content/Models -o OctaveNet8_LEVIR.pt  https://emailkntuacir-my.sharepoint.com/:u:/g/personal/farhadinima75_email_kntu_ac_ir/EWqQp2nWIiNIpZlf17NQMToBQRLgaWSFaZoy-6aaSbupbg?download=1")
  os.system("aria2c -x 8 -d /content/Models -o OctaveNet32_LEVIR.pt https://emailkntuacir-my.sharepoint.com/:u:/g/personal/farhadinima75_email_kntu_ac_ir/EaudSyHuOwpLuNCGdP9FiToB0tgQ50ZTLZykGouNk_Y0BQ?download=1")
  os.system("aria2c -x 8 -d /content/Models -o OctaveNet64_LEVIR.pt https://emailkntuacir-my.sharepoint.com/:u:/g/personal/farhadinima75_email_kntu_ac_ir/ESPVouHxSLlEn9s5Lw9sX1kBg_VEJOjoWHGqAnqYoAxPdQ?download=1")
  os.system("aria2c -x 8 -d /content/Models -o OctaveNet8_DSIFN.pt  https://emailkntuacir-my.sharepoint.com/:u:/g/personal/farhadinima75_email_kntu_ac_ir/EXKDGt0CI0NGp4bDvbAxeB8Bq4NZlC34-c_siXZ4GC9SkQ?download=1")
  os.system("aria2c -x 8 -d /content/Models -o OctaveNet32_DSIFN.pt https://emailkntuacir-my.sharepoint.com/:u:/g/personal/farhadinima75_email_kntu_ac_ir/EdHNCK09KcVGu-z6JD1AFmABG5EIWAeMSof6Uy7zxSfKLg?download=1")
  os.system("aria2c -x 8 -d /content/Models -o OctaveNet64_DSIFN.pt https://emailkntuacir-my.sharepoint.com/:u:/g/personal/farhadinima75_email_kntu_ac_ir/EW5Jjd9Gn0hDj8wxgeQBkWkBxWv9NcgWMeRaRfC6gW7qBA?download=1")
  os.system("aria2c -x 8 -d /content/Models -o OctaveNet8_CropLand.pt  https://emailkntuacir-my.sharepoint.com/:u:/g/personal/farhadinima75_email_kntu_ac_ir/Eeq1M8pXkWpOlF1c6DCK5vEByzA1E8gNFH6SRzSyrW-Zrw?download=1")
  os.system("aria2c -x 8 -d /content/Models -o OctaveNet32_CropLand.pt https://emailkntuacir-my.sharepoint.com/:u:/g/personal/farhadinima75_email_kntu_ac_ir/EbT77UrkJI1JsDOV9aKfLckBkiRXJ8SxXySYCKgTwpbPgg?download=1")
  os.system("aria2c -x 8 -d /content/Models -o OctaveNet64_CropLand.pt https://emailkntuacir-my.sharepoint.com/:u:/g/personal/farhadinima75_email_kntu_ac_ir/EQodTpsz3_BMrgbaIlkeg8UBZ-i3S-ltAdLn__ZfgF49-g?download=1")
  for i in glob.glob(f'{DatasetPath}/*.zip'): shutil.unpack_archive(i, DatasetPath)
  clear_output()
  print('Preparation was completed.')

def SeedAll(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
      torch.cuda.manual_seed(seed)
      torch.backends.cudnn.benchmark = False

class BaseDataset(torch.utils.data.Dataset):
    def __init__(self, Path, Phase='Train', Transformation=False):
      super(BaseDataset, self).__init__()
      self.ListsTXT = {'Train': np.loadtxt(f'{Path}/list/train.txt', dtype=str), 
                       'Val': np.loadtxt(f'{Path}/list/val.txt', dtype=str), 
                       'Test': np.loadtxt(f'{Path}/list/test.txt', dtype=str)}
      self.Phase = Phase
      self.Path = Path
      self.Transformation = Transformation
      if self.Phase == 'Train':
        Labels = np.array([cv2.imread(f'{Path}/label/{I}', -1) for I in tqdm.tqdm(self.ListsTXT['Train'])])
        TotalChanges = np.sum(Labels//255)
        TotalPixels = Labels.shape[0] * Labels.shape[1] * Labels.shape[2]
        TotalUnChanges = TotalPixels - TotalChanges
        self.Weights = torch.from_numpy(1 - np.array([(TotalUnChanges/TotalPixels)-0.25, (TotalChanges/TotalPixels)+0.25]))

    def __len__(self): return len(self.ListsTXT[self.Phase])

    def __getitem__(self, I):
      Image = self.ListsTXT[self.Phase][I]
      APatch = cv2.imread(f'{self.Path}/A/{Image}', -1)[..., ::-1].copy()
      BPatch = cv2.imread(f'{self.Path}/B/{Image}', -1)[..., ::-1].copy()
      Label = cv2.imread(f'{self.Path}/label/{Image}', -1)
      if self.Transformation:
        if random.random() > 0.5:
          APatch = APatch[::-1, :, :].copy()
          BPatch = BPatch[::-1, :, :].copy()
          Label = Label[::-1, :].copy()
        if random.random() > 0.5:
          APatch = APatch[:, ::-1, :].copy()
          BPatch = BPatch[:, ::-1, :].copy()
          Label = Label[:, ::-1].copy()
      APatch, BPatch = TF.to_tensor(APatch), TF.to_tensor(BPatch)
      return (TF.normalize(APatch, mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
              TF.normalize(BPatch, mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
              torch.from_numpy(Label) // 255)   

def Accuracy(input, target):
    return 100 * ((torch.count_nonzero(input == target)) / target.numel()).cpu().numpy()

def CompareOutputInColor(Pred, Label, SaveName):
  Pred, Label = Pred.astype('float32'), Label.astype('float32')
  ColorCM = np.zeros((Pred.shape[0], Pred.shape[1], 3), dtype='uint8')
  CM = Label - Pred # Values: -1, 0, 1
  TP = (Pred == 1) * (Label==1)
  for C in [-1, 0, 1]:
    if C != 0: Indexes = np.where(CM == C)
    else: Indexes = np.where(TP == True)
    if C == -1: Color = np.array([255, 10, 50])
    elif C == 0: Color = np.array([255, 255, 255])
    elif C == 1: Color = np.array([10, 10, 255])
    ColorCM[Indexes] = Color
  cv2.imwrite(SaveName, ColorCM)

def ComputeStatics(Pred, Label):
  TP, FP, FN, TN = get_stats(Pred, Label, mode='multiclass', num_classes=2)
  TP, FP = torch.sum(TP,dim=0), torch.sum(FP,dim=0)
  FN, TN = torch.sum(FN,dim=0), torch.sum(TN,dim=0)
  return torch.stack([TP, FP, FN, TN])

def ComputeMetrics(Statics):
  TP, FP, FN, TN = torch.sum(torch.stack(Statics), dim=0)
  P = precision(TP, FP, FN, TN, reduction=None).numpy()
  R = recall(TP, FP, FN, TN, reduction=None).numpy()
  F1 = 2 * (P * R) / (P + R)
  ACC = accuracy(TP, FP, FN, TN, reduction=None).numpy()
  IoU = iou_score(TP, FP, FN, TN, reduction=None).numpy()
  p1 = ((TP+FN) / (TP+FP+TN+FN)).numpy()
  p2 = ((TP+FP) / (TP+FP+TN+FN)).numpy()
  RandomACC = p1*p2 + (1-p1)*(1-p2)
  Kappa = (ACC - RandomACC) / (1 - RandomACC)
  return [ACC[1], Kappa[1], IoU[1], F1[1], P[1], R[1]]

def Test(DatasetDir, LoadStateDictPath, Dims, OptionalName,
       BatchSize, Classes, Device, ModelDir, Workers, Dataset):
  ModelName = f'OctaveNet{OptionalName}_{Dataset}'
  TestDS = BaseDataset(DatasetDir, Phase='Test')
  TestDL = torch.utils.data.DataLoader(TestDS, batch_size=BatchSize, num_workers=Workers, pin_memory=True)
  Net = OctaveNet(Dims, InCH=3, Classes=Classes).to(Device)
  Net.load_state_dict(torch.load(LoadStateDictPath, map_location=Device))
  Net.eval()
  Statics = []
  if os.path.isdir(f"{ModelDir}/{ModelName}"): shutil.rmtree(f"{ModelDir}/{ModelName}")
  os.makedirs(f"{ModelDir}/{ModelName}", exist_ok=True)
  for i, (PatchA, PatchB, Label) in tqdm.tqdm(enumerate(TestDL), total=len(TestDL), desc=f'Testing...'):
    with torch.inference_mode():
      PatchA, PatchB, Label = PatchA.float().to(Device), PatchB.float().to(Device), Label.long().to(Device)
      Pred = Net(PatchA, PatchB)
      Pred = torch.argmax(Pred, dim=1)
      Statics.append(ComputeStatics(Pred, Label))
    Pred, Label = Pred.cpu().numpy(), Label.cpu().numpy()
    for I in range(Pred.shape[0]):
      Acc = Accuracy(torch.from_numpy(Pred[I]), torch.from_numpy(Label[I]))
      cv2.imwrite(f"{ModelDir}/{ModelName}/{i}_{I}_Acc_{Acc:.4f}.png", Pred[I]*255)
      cv2.imwrite(f"{ModelDir}/{ModelName}/{i}_{I}.png", Label[I]*255)
      CompareOutputInColor(Pred[I], Label[I], f"{ModelDir}/{ModelName}/{i}_{I}_ColorCoded.png")
  Metrics = ComputeMetrics(Statics)
  with open(f"{ModelDir}/TestMetrics.txt", 'a') as F: 
    F.write(f"Acc: {Metrics[0]*100:.2f}  Kappa: {Metrics[1]*100:.2f}  IoU: {Metrics[2]*100:.2f}  F1: {Metrics[3]*100:.2f}  Precision: {Metrics[4]*100:.2f}  Recall: {Metrics[5]*100:.2f} \n")
  ZipFile = f"{ModelDir}/Output_{ModelName}.zip"
  if os.path.isfile(ZipFile): os.remove(ZipFile)
  shutil.make_archive(ZipFile[:-4], 'zip', ZipFile.split('_')[0])
  ReleaseMem()

def ReleaseMem():
  if Device == 'cuda': torch.cuda.empty_cache()
  gc.collect()
