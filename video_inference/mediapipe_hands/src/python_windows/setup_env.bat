python3.11 -m venv 311env
call .\311env\Scripts\activate.bat
call .\311env\Scripts\python -m pip install --upgrade pip wheel
call pip install "numpy~=1.26.4" opencv-python
