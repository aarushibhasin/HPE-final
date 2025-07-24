# SURREAL Dataset Download Guide

## 📋 **Step-by-Step Instructions**

### **Step 1: Get Access Credentials**

1. **Visit**: https://www.di.ens.fr/willow/research/surreal/data/
2. **Accept the license terms** and request access
3. **Wait for email** with your username and password

### **Step 2: Download the Dataset**

#### **Option A: Download Individual Files (Recommended)**

1. **Download the download script**:
   ```bash
   wget https://www.di.ens.fr/willow/research/surreal/data/download/download_surreal.sh
   ```

2. **Make it executable**:
   ```bash
   chmod +x download_surreal.sh
   ```

3. **Run the download script**:
   ```bash
   ./download_surreal.sh /path/to/surreal_dataset yourusername yourpassword
   ```

4. **For minimal subset, download only**:
   - `cmu/train/run0/` (a few sequences)
   - `cmu/val/run0/` (small subset)
   - `cmu/test/run0/` (small subset)

#### **Option B: Download Full Dataset (86GB)**

If you have storage space, download the full `SURREAL_v1.tar.gz` file.

### **Step 3: Extract the Dataset**

```bash
# If you downloaded the full tar.gz
tar -xzf SURREAL_v1.tar.gz

# The structure should be:
SURREAL/data/cmu/train/run0/01_01/
├── 01_01_c0001.mp4          # RGB video
├── 01_01_c0001_depth.mat    # Depth data
├── 01_01_c0001_segm.mat     # Segmentation
├── 01_01_c0001_info.mat     # Annotations (joints2D, joints3D, etc.)
```

### **Step 4: Convert to Our Format**

Use the conversion script to transform `.mat` files to JSON:

```bash
python scripts/convert_surreal_to_json.py \
    --surreal-root /path/to/SURREAL/data \
    --output-root ./data/surreal_3d \
    --max-frames 50
```

### **Step 5: Verify the Data**

```bash
python scripts/verify_dataset_3d.py
```

### **Step 6: Start Training**

```bash
python quick_start_3d.py --skip-data-download
```

## 🔧 **Alternative: Manual Download (Small Subset)**

If you want to download just a few sequences manually:

1. **Use the download script with specific paths**:
   ```bash
   ./download_surreal.sh /path/to/surreal_dataset yourusername yourpassword
   ```

2. **Download only specific sequences**:
   - `cmu/train/run0/01_01/`
   - `cmu/train/run0/01_02/`
   - `cmu/val/run0/01_01/`

## 📊 **Expected File Sizes**

- **Full SURREAL**: ~86GB
- **Minimal subset**: ~2-5GB (depending on sequences)
- **Converted JSON**: ~500MB

## 🐛 **Troubleshooting**

### **Download Issues**
- Check your credentials are correct
- Try downloading smaller subsets first
- Use a stable internet connection

### **Conversion Issues**
- Install required dependencies: `pip install scipy opencv-python`
- Check file permissions
- Verify `.mat` files are not corrupted

### **Memory Issues**
- Reduce `--max-frames` parameter
- Process one split at a time
- Use SSD storage for better performance

## 📞 **Need Help?**

If you encounter issues:
1. Check the SURREAL dataset documentation
2. Verify your credentials are working
3. Try downloading a smaller subset first
4. Contact the SURREAL dataset maintainers if needed 