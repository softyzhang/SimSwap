import PyInstaller.__main__

PyInstaller.__main__.run([
    'test_video_swapspecific_GUI1.py',
    '--noconfirm',
    '-w',
    '--log-level=DEBUG',
    '-i face.ico',
    '--hidden-import opencv-python',
    '--paths D:/Program_Files/opencv-4.6.0/opencv',
    '--upx-dir D:/Program_Files/upx-3.96-win64',
    '--exclude-module matplotlib',
    '--exclude-module qt5'
])