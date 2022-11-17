import os, shutil, glob
from IPython.display import clear_output
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
  print('Prepration done.')
