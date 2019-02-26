import cv2
import glob
import os

def movie_to_image(video_paths, out_image_path, num_cut=10):
    img_count = 0
    for video_path in video_paths:
        print(video_path)
        capture = cv2.VideoCapture(video_path)
        frame_count = 0
        while(capture.isOpened()):

            ret, frame = capture.read()
            if ret == False:
                break

            if frame_count % num_cut == 0:
                img_file_name = os.path.join(out_image_path, '{:05d}.jpg'.format(img_count))
                cv2.imwrite(img_file_name, frame)
                img_count += 1

            frame_count += 1

        capture.release()

if __name__ == '__main__':

    video_paths = glob.glob("./data/video/*.mp4")
    out_image_path = os.path.join("./data/images/neural-style/content", "yuyuta")

    print('Movie to image ...')
    movie_to_image(video_paths, out_image_path, num_cut=20)

    images = glob.glob(out_image_path + '/*.jpg')
    print('image num', len(images))
