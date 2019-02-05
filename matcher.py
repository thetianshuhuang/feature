

import matplotlib.pyplot as plt
import cv2
import random
import collections
from scipy import stats, optimize
import os

Match = collections.namedtuple("Match", ["target", "scene"])
MatchPair = collections.namedtuple("MatchPair", ["matches", "ratio"])


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

    def __search_kp(self, scene, task):
        s = task.subtask("Searching scene for target keypoints").start()
        kp, desc = self.akaze.detectAndCompute(scene, None)
        matches = self.matcher.knnMatch(desc, self.desc, 2)
        s.done(
            desc="Finished keypoint matcher; {n} points found"
            .format(n=len(matches)))

        return kp, matches

    def __ratio_test(self, kp, matches, task):
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

        return matches

    def __distance_ratio(self, sampled):
        m1, m2 = sampled

        target = (
            (m1.target[0] - m2.target[0])**2 +
            (m1.target[1] - m2.target[1])**2)
        scene = (
            (m1.scene[0] - m2.scene[0])**2 +
            (m1.scene[1] - m2.scene[1])**2)

        if target == 0 or scene == 0:
            return MatchPair(ratio=-1, matches=sampled)
        else:
            return MatchPair(ratio=(scene / target)**0.5, matches=sampled)

    def __distance_ratio_test(self, matches, task):
        s = task.subtask("Distance ratio test").start()

        # Large number of passing keypoints
        if len(matches) * (len(matches) - 1) / 2 > self.sample_size:
            task.print("Sampling {n} pairs".format(n=self.sample_size))
            sampled = [
                self.__distance_ratio(random.sample(matches, 2))
                for _ in range(self.sample_size)
            ]

        # Small number of passing keypoints
        else:
            task.print("Small number of matches -- using all pairs")
            sampled = [
                self.__distance_ratio((i, j))
                for i in matches for j in matches
            ]

        # Remove errored points (distance is zero)
        sampled = [i for i in sampled if i.ratio > 0]

        # Check for no points found
        if len(sampled) < 2:
            return [], []

        # Get scale
        distances = [i.ratio for i in sampled]
        k = stats.gaussian_kde(distances)
        scale = optimize.minimize(lambda x: - 1 * k(x), [0.5]).x[0]
        s.print("Scale: " + str(scale))

        # Get points satisfying scale condition
        matches_new = []
        target_range = [scale * 0.8, scale * 1.2]
        for pair in sampled:
            if target_range[0] < pair.ratio and pair.ratio < target_range[1]:
                matches_new += pair.matches

        s.done(desc="Finished distance histogram")

        return matches_new, distances

    def match(self, scene, task=None):

        task.start("AKAZE Keypoint Matcher: searching for " + self.name)

        kp, matches = self.__search_kp(scene, task)
        matches = self.__ratio_test(kp, matches, task)
        matches, distances = self.__distance_ratio_test(matches, task)

        task.done(desc="Done matching images")

        return matches, distances


def test_plot(akaze, img, task, hist=False):

    img = cv2.imread(img)
    matches, distances = akaze.match(
        cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), task=task)

    plt.figure(1)
    plt.subplot(121)
    plt.imshow(img)
    plt.scatter(
        [match.scene[0] for match in matches],
        [match.scene[1] for match in matches])
    for i, match in enumerate(matches):
        plt.annotate(str(i), match.scene)

    plt.subplot(122)
    plt.imshow(akaze.target)
    plt.scatter(
        [match.target[0] for match in matches],
        [match.target[1] for match in matches])
    for i, match in enumerate(matches):
        plt.annotate(str(i), match.target)

    if hist:
        plt.figure(2)
        plt.hist([i for i in distances if i < 2], bins=20)

    plt.show()


if __name__ == "__main__":
    import syllabus
    main = syllabus.Task("Test").start()

    m = AkazeMatcher(
        cv2.imread("../hx/ref.PNG"), task=main.subtask(), name="HeroX Gate")

    names = os.listdir("../hx/Data_Training")
    test_plot(
        m, "../hx/Data_Training/" + random.choice(names),
        main.subtask())

    main.done()
