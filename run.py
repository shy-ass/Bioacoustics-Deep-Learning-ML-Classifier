import os
import sys

def main():
    print("🚀 Booting up the Bioacoustics ML/DL Web Engine...")
    # This executes the streamlit command directly from python
    os.system(f"{sys.executable} -m streamlit run app.py")

if __name__ == "__main__":
    main()
