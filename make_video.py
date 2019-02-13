import argparse
import cv2
import os
from glob import glob
from tqdm import tqdm


def parse():
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input', type=str, default='.')
    parser.add_argument('-o', '--output', type=str, default='output.mp4')
    parser.add_argument('-ext', '--extension', type=str, default='jpg')
    parser.add_argument('-wh', '--with_header', action='store_true')
    return parser.parse_args()

def add_header(frame, text):
    frame = cv2.copyMakeBorder(frame, 60, 0, 0, 0, cv2.BORDER_CONSTANT)
    
    font                   = cv2.FONT_HERSHEY_SIMPLEX
    bottomLeftCornerOfText = (20, 40)
    fontScale              = 1
    fontColor              = (255, 255, 255)
    lineType               = 2

    cv2.putText(
        frame, 
        text, 
        bottomLeftCornerOfText, 
        font, 
        fontScale,
        fontColor,
        lineType
    )
    return frame

if __name__ == '__main__':
    args = parse()
    print(args)
    
    images = sorted(list(glob(os.path.join(args.input, '*.'+args.extension))))
    
    frame = cv2.imread(images[0])
    height, width, channels = frame.shape
    if args.with_header:
        height += 60
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(args.output, fourcc, 20.0, (width, height))
    
    for image in tqdm(images):
        frame = cv2.imread(image)
        if args.with_header:
            frame = add_header(frame, os.path.basename(image))
        out.write(frame)
        if (cv2.waitKey(1) & 0xFF) == ord('q'):
            break
    out.release()
    cv2.destroyAllWindows()
    print('Generated video', args.output)