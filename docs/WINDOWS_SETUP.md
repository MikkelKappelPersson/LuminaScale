# LuminaScale Windows Setup Guide

**For non-technical users** — This guide walks you through setting up and running LuminaScale on Windows.

---

## Step 1: Get the Project Code

### Option A: Download as ZIP (Easiest)

1. Go to the GitHub repository
2. Click the green **Code** button
3. Click **Download ZIP**
4. Extract the ZIP file to your computer (e.g., `C:\Users\YourName\Documents\LuminaScale`)

### Option B: Clone with Git (For Git Users)

If you have Git installed on your computer:

1. Open **Command Prompt** or **PowerShell**
2. Navigate to where you want the folder:
   ```
   cd C:\Users\YourName\Documents
   ```
3. Clone the repository:
   ```
   git clone https://github.com/MikkelKappelPersson/LuminaScale.git
   ```
4. Navigate into the folder:
   ```
   cd LuminaScale
   ```

---

## Step 2: Download Checkpoints

The model checkpoints are required to run inference.

1. Download the checkpoint files sent to you
2. Create or navigate to the `checkpoints/` folder in your LuminaScale directory
3. Place the checkpoint files inside this folder

Your folder structure should look like:
```
LuminaScale/
├── checkpoints/
│   ├── aces-mapper-20260425_231537-epoch=09.ckpt
│   ├── last.ckpt
│   └── (other checkpoint files here)
├── app.py
├── README.md
└── ...
```

---

## Step 3: Download and Install Pixi

Pixi is a package manager that handles all dependencies for you automatically.

1. Go to [pixi.sh](https://pixi.sh)
2. Click **Download** and select **Windows**
3. Follow the installation instructions on the website
4. After installation, **restart your terminal** (close and reopen Command Prompt or PowerShell)

To verify Pixi is installed, open a new terminal and run:
```
pixi --version
```

You should see a version number like `0.20.0` (exact number may differ).

---

## Step 4: Prepare Dependencies (Optional but Recommended)

This step is optional. When you run `pixi run`, it will automatically install dependencies if needed. However, you can pre-install them to speed things up:

1. Open **Command Prompt** or **PowerShell**
2. Navigate to the LuminaScale folder:
   ```
   cd path\to\LuminaScale
   ```
3. Run:
   ```
   pixi install
   ```

This will download and set up all required packages. It may take a few minutes.

---

## Step 5: Launch the Application

1. Open **Command Prompt** or **PowerShell**
2. Navigate to the LuminaScale folder:
   ```
   cd path\to\LuminaScale
   ```
3. Start the application:
   ```
   pixi run python .\app.py
   ```

   **First time?** It may take 30-60 seconds to start (warming up GPU). Just wait.

4. Look for this message in the terminal:
   ```
   Running on local URL: http://127.0.0.1:7860
   ```

5. Open your web browser and go to: **http://127.0.0.1:7860**

6. You should see the LuminaScale interface! Upload an image and start processing.

### To Stop the Application
Press **Ctrl+C** in the command prompt.

---

## First Inference Run — What to Expect

When you run inference **for the first time**, the application needs to generate a Look-Up Table (LUT), which is an internal optimization. This takes extra time:

- **First run:** 2-10 minutes (generating LUT)
- **Subsequent runs:** 5 -30 seconds (much faster!)

**This is normal.** Just be patient during the first run. The terminal will show progress messages. Once complete, future inference runs will be significantly faster.

You'll see messages like:
```
Generating LUT...
LUT generation complete.
Running inference...
```

After the first run, results will appear much quicker! ✓

---

## Troubleshooting

### "Pixi is not installed"
- Make sure you restarted your terminal after installing Pixi
- Try running `pixi --version` to confirm

### "Checkpoint files not found"
- Make sure checkpoint files are in the `checkpoints/` folder
- Check file names match exactly: `aces-mapper-*.ckpt` and `last.ckpt`

### "CUDA not available" or "GPU not detected"
- The app will fall back to CPU (slower but still works)
- To use GPU, make sure NVIDIA drivers are installed

### Application won't start
- Check the terminal for error messages
- Make sure you're in the LuminaScale directory
- Try running `pixi install` first, then `pixi run python .\app.py`

---

## Next Steps

Once you have the app running:
- **Upload an image** — Click "Upload input image" or select from the gallery
- **Configure settings** — Adjust parameters like crop size and alignment
- **Run inference** — Click "Run full inference"
- **Download results** — Select outputs and click "Prepare Download"

