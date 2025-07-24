#!/usr/bin/env python3
"""
Windows-compatible SURREAL dataset downloader
"""
import os
import sys
import argparse
import requests
import zipfile
from pathlib import Path
from tqdm import tqdm

def download_file_with_auth(url, filename, username, password, description="Downloading"):
    """Download file with authentication and progress bar"""
    try:
        # Create session with authentication
        session = requests.Session()
        session.auth = (username, password)
        
        # Stream the download
        response = session.get(url, stream=True)
        response.raise_for_status()
        
        # Get file size
        total_size = int(response.headers.get('content-length', 0))
        
        # Download with progress bar
        with open(filename, 'wb') as f, tqdm(
            desc=description,
            total=total_size,
            unit='B',
            unit_scale=True,
            unit_divisor=1024,
        ) as pbar:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
                    pbar.update(len(chunk))
        
        return True
    except Exception as e:
        print(f"❌ Error downloading {url}: {e}")
        return False

def download_surreal_subset(output_dir, username, password, sequences=None):
    """Download a subset of SURREAL dataset"""
    
    if sequences is None:
        # Default minimal subset
        sequences = [
            "cmu/train/run0/01_01",
            "cmu/train/run0/01_02", 
            "cmu/val/run0/01_01",
            "cmu/test/run0/01_01"
        ]
    
    base_url = "https://lsh.paris.inria.fr/SURREAL/data"
    
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    successful_downloads = 0
    total_downloads = 0
    
    for sequence in sequences:
        print(f"\n📥 Downloading sequence: {sequence}")
        
        # Create sequence directory
        seq_dir = output_path / sequence
        seq_dir.mkdir(parents=True, exist_ok=True)
        
        # Try to download the sequence files
        # Note: This is a simplified approach - you may need to adjust based on actual SURREAL structure
        
        # Try different file patterns
        file_patterns = [
            f"{sequence.split('/')[-1]}_c0001.mp4",
            f"{sequence.split('/')[-1]}_c0001_info.mat",
            f"{sequence.split('/')[-1]}_c0001_depth.mat"
        ]
        
        for pattern in file_patterns:
            url = f"{base_url}/{sequence}/{pattern}"
            local_file = seq_dir / pattern
            
            total_downloads += 1
            
            if not local_file.exists():
                if download_file_with_auth(url, local_file, username, password, f"Downloading {pattern}"):
                    successful_downloads += 1
                    print(f"✅ Downloaded: {pattern}")
                else:
                    print(f"⚠️  Failed to download: {pattern}")
            else:
                print(f"✅ File already exists: {pattern}")
                successful_downloads += 1
    
    print(f"\n📊 Download Summary:")
    print(f"   Total files attempted: {total_downloads}")
    print(f"   Successfully downloaded: {successful_downloads}")
    print(f"   Success rate: {successful_downloads/total_downloads*100:.1f}%")
    
    return successful_downloads > 0

def main():
    parser = argparse.ArgumentParser(description="Download SURREAL dataset subset (Windows compatible)")
    parser.add_argument('--output-dir', default='./data/surreal_raw',
                       help='Output directory for downloaded files')
    parser.add_argument('--username', required=True,
                       help='SURREAL dataset username')
    parser.add_argument('--password', required=True,
                       help='SURREAL dataset password')
    parser.add_argument('--sequences', nargs='+',
                       help='Specific sequences to download')
    
    args = parser.parse_args()
    
    print("🚀 SURREAL Dataset Download (Windows)")
    print("=" * 50)
    
    # Download subset
    success = download_surreal_subset(args.output_dir, args.username, args.password, args.sequences)
    
    if success:
        print(f"\n✅ Download complete!")
        print(f"📁 Files downloaded to: {args.output_dir}")
        print("\n📋 Next steps:")
        print("   1. Convert .mat files to JSON format")
        print("   2. Run: python scripts/convert_surreal_to_json.py --surreal-root ./data/surreal_raw")
        print("   3. Start training: python quick_start_3d.py --skip-data-download")
    else:
        print("\n❌ Download failed!")
        print("Please check your credentials and try again.")

if __name__ == "__main__":
    main() 