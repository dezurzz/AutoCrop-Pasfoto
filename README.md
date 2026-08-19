# Web Application - AutoCrop Pasfoto (3x4 Face-Aware Cropper)

This folder contains the **Streamlit web application** for the AutoCrop Pasfoto utility, enabling a cross-platform user-friendly GUI.

## Features
- **Dynamic Fine-Tuning**: Real-time sliders to customize face ratio, position, background color, and manual fine offsets (rotation, x-offset, y-offset) with live feedback.
- **Batch Processing**: Upload multiple photos, process them simultaneously, and download all cropped results as a `.zip` archive.
- **Zero Configuration**: Automated model downloader fetches required model files when started.

---

## Local Setup & Run

1. Navigate to the `web_app` directory:
   ```bash
   cd web_app
   ```

2. Create a virtual environment and activate it:
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```

3. Install required packages:
   ```bash
   pip install -r requirements.txt
   ```

4. Launch the Streamlit application:
   ```bash
   streamlit run app.py
   ```

5. Access the app in your browser at `http://localhost:8501`.

---

## Deployment to Streamlit Community Cloud (Free)

1. **Commit and Push to GitHub**:
   Push this repository (or just this folder structure) to your public GitHub account.

2. **Sign up at Streamlit Cloud**:
   Go to [share.streamlit.io](https://share.streamlit.io/) and log in using your GitHub account.

3. **Deploy App**:
   - Click **"Create app"** (or "New app").
   - Select your Repository, Branch, and specify the Main file path as `web_app/app.py` (or `app.py` if the repository root is this folder).
   - Streamlit Cloud will automatically detect `requirements.txt` and `packages.txt` to provision and install all dependencies.
   - Within 2-3 minutes, your app will be online and accessible via a public URL (e.g. `https://your-app-name.streamlit.app`).
# AutoCrop-Pasfoto
