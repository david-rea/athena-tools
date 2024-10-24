"""
Function for making animations
"""

import cv2


def animate(files, filename, fps=10, **kwargs):
    """
    Animate a a collection of still images
    : files    : list of image files that make the individual movie frames
    : filename : the animation will save to 'filename.mp4'
    : fps      : frames per second
    keyword arguments are passed to 'cv2.VideoWriter'
    """
        
    if filename[-4] not '.mp4':
        filename += '.mp4'
    
    w = None
    for file in files:

        frame = cv2.imread(file)

        if w is None: # first iteration, set up writer
            h, w, _ = frame.shape
            fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
            writer = cv2.VideoWriter(filename, fourcc, fps, (w,h), **kwargs)

        writer.write(frame)

        # if you want to extend the frame
        # for repeat in range(duration*fps):
        #     writer.write(frame)

    writer.release()
    
    print(f"Video file '{filename}' created")
