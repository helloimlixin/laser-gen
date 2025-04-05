#!/usr/bin/env python
"""
Script to download and prepare datasets for VQ-GAN training.
Some datasets require manual downloads or have specific preparation steps.
"""

import os
import argparse
import subprocess
import shutil
from pathlib import Path
import zipfile
import tarfile
import gzip
import requests
from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser(description='Download and prepare datasets for VQ-GAN')
    parser.add_argument('--dataset', type=str, required=True,
                        choices=['coco', 'lsun', 'div2k', 'celeba_hq', 'ffhq'],
                        help='Dataset to download')
    parser.add_argument('--data_dir', type=str, default='data',
                        help='Directory to store datasets')
    return parser.parse_args()


def download_file(url, filepath):
    """Download a file from a URL to a local filepath with progress bar"""
    response = requests.get(url, stream=True)
    total_size = int(response.headers.get('content-length', 0))
    block_size = 1024  # 1 Kibibyte
    
    with open(filepath, 'wb') as f, tqdm(
            desc=f"Downloading {os.path.basename(filepath)}",
            total=total_size,
            unit='B',
            unit_scale=True,
            unit_divisor=1024,
        ) as pbar:
        for data in response.iter_content(block_size):
            f.write(data)
            pbar.update(len(data))


def extract_archive(filepath, extract_dir, archive_format=None):
    """Extract an archive file to the specified directory"""
    if archive_format is None:
        if filepath.endswith('.zip'):
            archive_format = 'zip'
        elif filepath.endswith('.tar'):
            archive_format = 'tar'
        elif filepath.endswith('.tar.gz') or filepath.endswith('.tgz'):
            archive_format = 'tar.gz'
        elif filepath.endswith('.gz') and not filepath.endswith('.tar.gz'):
            archive_format = 'gz'
        else:
            raise ValueError(f"Unsupported archive format: {filepath}")
    
    print(f"Extracting {filepath} to {extract_dir}...")
    
    if archive_format == 'zip':
        with zipfile.ZipFile(filepath, 'r') as zip_ref:
            zip_ref.extractall(extract_dir)
    
    elif archive_format in ['tar', 'tar.gz']:
        mode = 'r:gz' if archive_format == 'tar.gz' else 'r'
        with tarfile.open(filepath, mode) as tar_ref:
            tar_ref.extractall(extract_dir)
    
    elif archive_format == 'gz':
        output_filepath = os.path.join(extract_dir, os.path.basename(filepath)[:-3])
        with gzip.open(filepath, 'rb') as f_in:
            with open(output_filepath, 'wb') as f_out:
                shutil.copyfileobj(f_in, f_out)
    
    print(f"Extraction complete")


def download_coco(data_dir):
    """Download and prepare the COCO dataset"""
    coco_dir = os.path.join(data_dir, 'coco')
    os.makedirs(coco_dir, exist_ok=True)
    
    # URLs for COCO 2017
    urls = {
        'train_images': 'http://images.cocodataset.org/zips/train2017.zip',
        'val_images': 'http://images.cocodataset.org/zips/val2017.zip',
        'annotations': 'http://images.cocodataset.org/annotations/annotations_trainval2017.zip',
    }
    
    # Download and extract each part
    for name, url in urls.items():
        filepath = os.path.join(coco_dir, os.path.basename(url))
        
        if not os.path.exists(filepath):
            print(f"Downloading {name} from {url}")
            download_file(url, filepath)
        else:
            print(f"{name} already downloaded at {filepath}")
        
        extract_archive(filepath, coco_dir)
    
    print("COCO dataset preparation complete")


def download_lsun(data_dir):
    """Download and prepare the LSUN dataset"""
    lsun_dir = os.path.join(data_dir, 'lsun')
    os.makedirs(lsun_dir, exist_ok=True)
    
    print("LSUN dataset requires manual preparation:")
    print("1. Clone the LSUN repository:")
    print("   git clone https://github.com/fyu/lsun.git")
    print("2. Download the LSUN dataset using the provided scripts:")
    print("   python lsun/download.py -c bedroom")
    print("3. Prepare the LSUN data:")
    print("   python lsun/data.py export <lsun-data-folder> --out_dir data/lsun")
    print("\nSee https://github.com/fyu/lsun for detailed instructions")
    
    # Optionally provide a script that automates this process
    answer = input("Would you like to automatically download the LSUN bedroom category? (y/n): ")
    if answer.lower() == 'y':
        # Clone LSUN repository
        if not os.path.exists("lsun-repo"):
            subprocess.run(["git", "clone", "https://github.com/fyu/lsun.git", "lsun-repo"])
        
        # Install required packages
        subprocess.run(["pip", "install", "lmdb", "opencv-python"])
        
        # Download LSUN bedroom
        subprocess.run(["python", "lsun-repo/download.py", "-c", "bedroom"])
        
        # Process the data
        subprocess.run(["python", "lsun-repo/data.py", "export", 
                      "bedroom_train_lmdb", "--out_dir", os.path.join(lsun_dir, "bedroom_train")])
        subprocess.run(["python", "lsun-repo/data.py", "export", 
                      "bedroom_val_lmdb", "--out_dir", os.path.join(lsun_dir, "bedroom_val")])
        
        print("LSUN bedroom dataset preparation complete")
    else:
        print("Please follow the manual instructions to download and prepare the LSUN dataset")


def download_div2k(data_dir):
    """Download and prepare the DIV2K dataset"""
    div2k_dir = os.path.join(data_dir, 'DIV2K')
    os.makedirs(div2k_dir, exist_ok=True)
    
    # URLs for DIV2K
    urls = {
        'train_hr': 'http://data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K_train_HR.zip',
        'valid_hr': 'http://data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K_valid_HR.zip',
    }
    
    # Download and extract each part
    for name, url in urls.items():
        filepath = os.path.join(div2k_dir, os.path.basename(url))
        
        if not os.path.exists(filepath):
            print(f"Downloading {name} from {url}")
            download_file(url, filepath)
        else:
            print(f"{name} already downloaded at {filepath}")
        
        extract_archive(filepath, div2k_dir)
    
    print("DIV2K dataset preparation complete")


def download_celeba_hq(data_dir):
    """Provide instructions for downloading and preparing the CelebA-HQ dataset"""
    celeba_hq_dir = os.path.join(data_dir, 'celeba_hq')
    os.makedirs(celeba_hq_dir, exist_ok=True)
    
    print("CelebA-HQ dataset requires manual download:")
    print("1. Follow the instructions at:")
    print("   https://github.com/tkarras/progressive_growing_of_gans")
    print("2. Alternatively, download from:")
    print("   https://github.com/switchablenorms/CelebAMask-HQ")
    print("   (this version includes the original CelebA-HQ images)")
    print(f"3. Place the processed images in {celeba_hq_dir}")
    
    # Since no direct download links are available, we can't automate this process
    print("\nNote: The CelebA-HQ dataset requires login credentials and manual processing.")
    print("Please follow the instruction links to obtain the dataset.")


def download_ffhq(data_dir):
    """Provide instructions for downloading and preparing the FFHQ dataset"""
    ffhq_dir = os.path.join(data_dir, 'ffhq')
    os.makedirs(ffhq_dir, exist_ok=True)
    
    print("FFHQ dataset requires manual download:")
    print("1. Visit the official repository at:")
    print("   https://github.com/NVlabs/ffhq-dataset")
    print("2. Follow the instructions to download the dataset")
    print("   - You can download the dataset at different resolutions")
    print("   - For VQ-GAN training, 256×256 images are sufficient to start with")
    print(f"3. Place the images in {ffhq_dir}")
    
    print("\nAlternatively, you can use the download script from the FFHQ repository:")
    print("1. Clone the repository: git clone https://github.com/NVlabs/ffhq-dataset.git")
    print("2. Install the required dependencies")
    print("3. Run the download script: python download_ffhq.py --images --thumbnails")
    
    print("\nNote: The full FFHQ dataset is quite large (around 70GB for full resolution).")
    print("The thumbnails (256×256) are about 1.8GB and work well for initial experiments.")

def main():
    args = parse_args()
    
    # Create the data directory
    os.makedirs(args.data_dir, exist_ok=True)
    
    # Download the requested dataset
    if args.dataset == 'coco':
        download_coco(args.data_dir)
    elif args.dataset == 'lsun':
        download_lsun(args.data_dir)
    elif args.dataset == 'div2k':
        download_div2k(args.data_dir)
    elif args.dataset == 'celeba_hq':
        download_celeba_hq(args.data_dir)
    elif args.dataset == 'ffhq':
        download_ffhq(args.data_dir)


if __name__ == "__main__":
    main()
