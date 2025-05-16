# USD to GLB Converter

This tool converts USD (Universal Scene Description) files to GLB (GL Transmission Format Binary) format using either usdcat (if available) or Blender.

## Requirements

- Python 3.6+
- Either:
  - usdcat (part of the USD toolkit)
  - Blender (with USD import and GLB export capabilities)

## Installation

No additional Python packages are required for this script.

If you don't have usdcat installed, you'll need Blender:

- Download and install Blender from [blender.org](https://www.blender.org/download/)
- Make sure Blender is in your PATH or specify its location using the `--blender-path` argument

## Usage

```bash
# Activate the required conda environment first
conda activate base

# Run the converter script
python usd_to_glb_converter.py --input /path/to/your/model.usd --output /path/to/save/model.glb
```

### Command-line Arguments

- `--input` or `-i`: Path to the input USD file (required)
- `--output` or `-o`: Path to save the output GLB file (optional, defaults to the same path as input with .glb extension)
- `--blender-path`: Path to the Blender executable (optional, will try to find automatically)

### Example

```bash
conda activate base
python usd_to_glb_converter.py --input ../assets/model.usd --output ../converted/model.glb
```

## How It Works

The script attempts to convert the USD file to GLB format using the following methods, in order:

1. First, it tries to use `usdcat` if it's available on your system
2. If `usdcat` is not available or fails, it tries to use Blender:
   - It automatically locates Blender on your system
   - It creates a Python script for Blender to execute
   - It runs Blender in background mode to perform the conversion
   - It monitors the conversion process and reports any errors

## Notes

- The conversion process may take some time depending on the complexity of the USD file
- No internet connection is required as the conversion happens locally
- If the conversion fails with both methods, detailed error messages will be provided

## How to use rerun

```python
python tools/statistics/rerun_path_hdf5.py --data-dir /ssd/gello_software/gello_pd_dataset --color-by-task --save-to ./test_rerun.rrd --position-only
```
