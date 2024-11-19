sudo apt update
sudo apt install -y python3-pip git
git clone https://github.com/carbondirect/cd_atlite
cd cd_atlite/
sudo apt install python3.11-venv
python3 -m venv venv
source venv/bin/activate
pip install -e .
pip install pip-tools
pip-compile pyproject.toml 
pip install -r requirements.txt 
python wind_cf.py 


# extra disk stuff
   55  lsblk
   56  sudo mkfs.ext4 -m 0 -F -E lazy_itable_init=0,lazy_journal_init=0,discard /dev/sda
   57  sudo mkdir -p /mnt/disks/era5-disk
   58  sudo mount -o discard,defaults /dev/sda /mnt/disks/era5-disk
   59  sudo nano /etc/fstab
   60  ls -al /mnt/disks/era5-disk/
   61  ls
   62  export DASK_TEMPORARY_DIRECTORY="/mnt/disks/era5-disk/tmp"
   63  mkdir -p /mnt/disks/era5-disk/tmp
   64  sudo mkdir -p /mnt/disks/era5-disk/tmp
   65  chmod a+w /mnt/disks/era5-disk/tmp
   66  sudo chmod a+w /mnt/disks/era5-disk/tmp
   67  history
