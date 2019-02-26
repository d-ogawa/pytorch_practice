from pytube import YouTube
import os
import glob

def download_video(link, save_dir):
    print('Download %s to %s ...' %(link, save_dir))
    YouTube(link).streams.first().download(save_dir)
    # YouTube(link).streams.first().download()
    print('Done\n')

link = "https://www.youtube.com/watch?v=0MqlNBcUw5I"
save_dir = "./data/video"
download_video(link, save_dir)

videos = glob.glob(os.path.join(save_dir, '*.mp4'))
for (n, video) in enumerate(videos):
    os.rename(video, os.path.join(save_dir, 'video-{:02}.mp4'.format(n)))
