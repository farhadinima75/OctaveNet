import os, shutil, glob
from IPython.display import clear_output
def PrepareLibrariesAndDatasets():
  os.system("pip install torch==1.11.0+cu113 torchvision==0.12.0+cu113 torchaudio==0.11.0 --extra-index-url https://download.pytorch.org/whl/cu113")
  os.system("apt update")
  os.system("apt install aria2")
  DatasetPath = '/content/Datasets'")
  os.makedirs(DatasetPath, exist_ok=True)
  os.system("aria2c -x 8 -d /content/Datasets -o LEVIR-CD-256.zip    https://emailkntuacir-my.sharepoint.com/:u:/g/personal/farhadinima75_email_kntu_ac_ir/ERDI-5yFHahDl-1a390i2DgBJULY_BNh6vctHm4swinNZg?download=1")
  os.system("aria2c -x 8 -d /content/Datasets -o DSIFN-CD-256.zip    https://emailkntuacir-my.sharepoint.com/:u:/g/personal/farhadinima75_email_kntu_ac_ir/EeMlQQld78pGgCq3puurff0B8yrzo0qkN1-q0DfdpLDCFw?download=1")
  os.system("aria2c -x 8 -d /content/Datasets -o WHU-CD-256.zip      https://emailkntuacir-my.sharepoint.com/:u:/g/personal/farhadinima75_email_kntu_ac_ir/Ebh6rPs8y1xAi4In2D8Crc4BvRIy7_kSUj7EvNRfPcAKyQ?download=1")
  os.system("aria2c -x 8 -d /content/Datasets -o CropLand-CD-256.zip https://emailkntuacir-my.sharepoint.com/:u:/g/personal/farhadinima75_email_kntu_ac_ir/EVUpqzbgHF9Clr8pdNdaorQBWJQnDaojrHkcNDTBmRIojw?download=1")
  for i in glob.glob(f'{DatasetPath}/*.zip'): shutil.unpack_archive(i, DatasetPath)
  clear_output()
