# Digital Image Processing – Project 1

**Course:** Digital Image Processing  
**Institution:** Harokopio University of Athens  
**Academic Year:** 2025  
**Author:** Despina Ioanna Chalkiadaki  

---

## 📦 Requirements

- Python **3.13.2**
- Virtual environment (recommended)
- The following Python libraries:
  - `os` – For basic filesystem operations
  - `numpy` – For numerical computations and array processing
  - `pandas` – For handling and analyzing tabular data (e.g., CSV files)
  - `matplotlib.pyplot` – For visualizing images and creating plots
  - `scikit-learn` – Used only in Exercise 1 for K-means clustering
  - `scikit-image` – For basic and advanced image processing functions (transformations, filters, etc.)
  - `opencv-python` – For image segmentation and processing (a more specialized alternative to matplotlib)

---

## ⚙️ Setup Instructions

### 1. Install Python 3.13.2

Download and install Python 3.13.2 from the official website:  
👉 https://www.python.org/downloads/release/python-3132/

> If you're using `pyenv`, you can install it with:
> ```bash
> pyenv install 3.13.2
> pyenv local 3.13.2
> ```

---

### 2. Create a Virtual Environment

From your project root directory:

```bash
python3 -m venv venv
source venv/bin/activate       # macOS/Linux
venv\Scripts\activate          # Windows
