import matplotlib.pyplot as plt
import numpy as np
#from gym.wrappers import LazyFrames

from moviepy.editor import VideoClip


def loss_plot(losses, file, title=None):
    if title is None:
        title = 'Loss'
    log_scale = np.min(losses) > 0
    simple_plot(losses, title=title, show=False, file=file, ylabel='Loss', xlabel='Iters', log=log_scale)


def simple_plot(data, std=None, title=None, show=False, file=None, ylabel=None, xlabel=None, log=False):
    plt.figure()
    if log:
        plt.semilogy(data)
    else:
        plt.plot(data)

    if std is not None:
        assert not log, 'not sure how to implement this with log'
        upper = np.add(data, std)
        lower = np.subtract(data, std)
        xs = np.arange(len(lower))
        plt.fill_between(xs, lower, upper, alpha=0.3)

    if title is not None:
        plt.title(title)

    if xlabel is not None:
        plt.xlabel(xlabel)
    if ylabel is not None:
        plt.ylabel(ylabel)

    if file is not None:
        plt.savefig(file)

    if show:
        plt.show()
    else:
        plt.close()


def make_movie(trajectory, file, speedup=1):
    fps = 24
    def float_to_int(im):
        if np.max(im) <= 1:
            im = im * 255
            im = im.astype(int)
        im = np.nan_to_num(im)
        return im
    ims = []
    for frame in trajectory[::speedup]:
        if type(frame) == dict:
            frame = frame['obs']
        #if type(frame) == LazyFrames:
         #   frame = frame[0]
        if type(frame) == np.ndarray:
            if frame.shape[0] == 3:
                frame = frame.transpose((1, 2, 0))
            ims.append(float_to_int(frame))
            # print(float_to_int(frame.transpose((1, 2, 0))).shape)
        else:
            raise ValueError('Type {0} invalid'.format(type(frame)))

    def make_frame(t):
        """Returns an image of the frame for time t."""
        # ... create the frame with any library here ...
        return ims[int(round(t*fps))]

    if 'gif' in file:
        codec = 'gif'
    elif 'mp4' in file:
        codec = 'mpeg4'
    else:
        codec = None

    duration = int(np.ceil(len(ims) / fps))
    while len(ims) < duration * fps + 1:
        ims.append(ims[-1])
    clip = VideoClip(make_frame, duration=duration)
    clip.set_fps(fps)
    clip.write_videofile(file, fps=fps, codec=codec, verbose=False, logger=None)

