

import matplotlib.pyplot as plt
import cv2
import random
import collections
from scipy import stats, optimize

Match = collections.namedtuple("Match", ["target", "scene"])


class AkazeMatcher:

    def __init__(
            self, target, ratio=0.8, task=None,
            name="Generic Target", samples=100):

        task.start()

        self.akaze = cv2.AKAZE_create()
        self.matcher = cv2.DescriptorMatcher_create(
            cv2.DescriptorMatcher_BRUTEFORCE_HAMMING)

        self.name = name
        self.ratio = ratio
        self.sample_size = samples

        self.target = target
        self.kp, self.desc = self.akaze.detectAndCompute(
            cv2.cvtColor(self.target, cv2.COLOR_BGR2GRAY), None)

        task.done()

    def __distance_ratio(self, sampled):
        m1, m2 = sampled

        target = (
            (m1.target[0] - m2.target[0])**2 +
            (m1.target[1] - m2.target[1])**2)
        scene = (
            (m1.scene[0] - m2.scene[0])**2 +
            (m1.scene[1] - m2.scene[1])**2)

        if target == 0:
            return 0
        else:
            return (scene / target)**0.5

    def match(self, scene, task=None):

        task.start("AKAZE Keypoint Matcher: searching for " + self.name)

        s = task.subtask("Searching scene for target keypoints").start()
        kp, desc = self.akaze.detectAndCompute(scene, None)
        matches = self.matcher.knnMatch(desc, self.desc, 2)
        s.done(
            desc="Finished keypoint matcher; {n} points found"
            .format(n=len(matches)))

        s = task.subtask("Running ratio test").start()
        matches = [
            Match(
                target=self.kp[m.trainIdx].pt,
                scene=kp[m.queryIdx].pt)
            for m, n in matches
            if m.distance < self.ratio * n.distance]
        s.done(
            desc="Finished ratio test; {n} points passed"
            .format(n=len(matches)))

        s = task.subtask("Distance ratio test").start()
        # Large number of passing keypoints
        if len(matches) * (len(matches) - 1) / 2 > self.sample_size:
            task.print("Sampling {n} pairs".format(n=self.sample_size))
            distances = [
                self.__distance_ratio(random.sample(matches, 2))
                for _ in range(self.sample_size)
            ]
        # Small number of passing keypoints
        else:
            task.print("Small number of matches -- using all pairs")
            distances = [
                self.__distance_ratio(i, j)
                for i in matches for j in matches
            ]
        # Remove zeros
        distances = [i for i in distances if 0 < i and i < 2.5]
        k = stats.gaussian_kde(distances)
        s.print(optimize.minimize(lambda x: - 1 * k(x), [0.5]).x)
        s.done(desc="Finished distance histogram")

        task.done(desc="Done matching images")

        return matches, distances


def test_plot(akaze, img, task):

    img = cv2.imread(img)
    matches, distances = akaze.match(
        cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), task=task)

    plt.figure(1)
    plt.imshow(img)
    plt.scatter(
        [match.scene[0] for match in matches],
        [match.scene[1] for match in matches])
    for i, match in enumerate(matches):
        plt.annotate(str(i), match.scene)

    plt.figure(2)
    plt.imshow(akaze.target)
    plt.scatter(
        [match.target[0] for match in matches],
        [match.target[1] for match in matches])
    for i, match in enumerate(matches):
        plt.annotate(str(i), match.target)

    plt.figure(3)
    plt.hist(distances, bins=20)

    plt.show()


if __name__ == "__main__":
    import syllabus
    main = syllabus.Task("Test").start()

    m = AkazeMatcher(cv2.imread("../hx/ref.PNG"), task=main.subtask())
    test_plot(m, "../hx/Data_Training/IMG_0060.JPG", main.subtask())

    main.done()
