"""
Function for making animations

Install cv2 with
> pip install opencv-python
"""

import cv2


def progress(it: int, total: int):

    fill = "█"
    length = 50

    fraction = it/total
    filledLength = int(length*fraction)
    
    bar = fill*filledLength + "-"*(length - filledLength)    

    print(f"\rProgress |{bar}| {100*fraction:.1f}% Complete", end='\r')

def animate(images, outfile, fps=10, **kwargs):
    """
    Animate a a collection of still images
    : images  : list of image files that make the individual movie frames (using e.g. 'glob' module)
    : outfile : the animation will save to 'outfile.mp4'
    : fps     : desired frames per second
    keyword arguments are passed to 'cv2.VideoWriter'
    """
    
    if outfile[-4] != ".mp4":
        outfile += ".mp4"

    N = len(images)
    
    w = None
    for i, image in enumerate(images):

        frame = cv2.imread(image)

        if w is None: # first iteration, set up writer
            h, w, _ = frame.shape
            fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
            writer = cv2.VideoWriter(outfile, fourcc, fps, (w,h), **kwargs)

        writer.write(frame)

        # if you want to extend the frame
        #
        # for repeat in range(duration*fps):
        #     writer.write(frame)
        #

        progress(i+1, N)

    writer.release()
    
    print(f"\nVideo file '{outfile}' created")
