
import cv2
import random
from scipy import stats, optimize

import collections
Match = collections.namedtuple("Match", ["target", "scene"])
MatchPair = collections.namedtuple("MatchPair", ["matches", "ratio"])


class AkazeMatcher:

    """AKAZE based feature matcher

    Parameters
    ----------
    target : np.array
        Target image for the AKAZE matcher
    ratio : float
        Ratio test threshold parameter
    task : Task object
        Task to register under
    name : str
        Name of the target
    samples : int
        Number of samples for the distance ratio histogram

    Attributes
    ----------
    __akaze : cv2.AKAZE object
        Akaze matcher engine
    __matcher : cv2.DescriptorMatcher
        Descriptor matcher engine
    __name : str
        Target name
    __ratio : float
        Ratio test threshold parameter
    __sample_size : int
        Number of samples for the distance ratio histogram
    target : np.array
        Target image; if RGB, is converted to grayscale when fed into
        AKAZE.detectAndCompute.
    """

    def __init__(
            self, target, ratio=0.8, task=None,
            name="Generic Target", samples=100):

        s = task.subtask(
            "AKAZE",
            desc="Initializing AKAZE matcher for " + name).start()

        self.__akaze = cv2.AKAZE_create()
        self.__matcher = cv2.DescriptorMatcher_create(
            cv2.DescriptorMatcher_BRUTEFORCE_HAMMING)

        self.__name = name
        self.__ratio = ratio
        self.__sample_size = samples

        self.target = target
        self.__kp, self.__desc = self.__akaze.detectAndCompute(
            self.target if len(self.target.shape) == 2 else
            cv2.cvtColor(self.target, cv2.COLOR_BGR2GRAY),
            None)

        s.done(
            desc="Created AKAZE matcher; {n} keypoints created."
            .format(n=len(self.__kp)))

    def __search_kp(self, scene, task):
        """Search scene for target keypoints

        Parameters
        ----------
        scene : np.array
            Scene image to search for keypoints in
        task : Task object
            Parent task to create subtask for

        Returns
        -------
        [kp, matches]
            [List of keypoints, List of matches]
        """
        s = task.subtask(
            "KP Identify", desc="Identifying keypoints in scene").start()
        kp, desc = self.__akaze.detectAndCompute(scene, None)
        s.done(desc="Finished identifying points in scene.")

        s = task.subtask(
            "KP Match",
            desc="Matching scene keypoints to target keypoints").start()
        matches = self.__matcher.knnMatch(desc, self.__desc, 2)
        s.done(
            desc="Finished keypoint matcher; {n} matches found."
            .format(n=len(matches)))

        return kp, matches

    def __ratio_test(self, kp, matches, task):
        """Ratio test for keypoint matches

        Parameters
        ----------
        kp : keypoints[]
            List of keypoints
        matches : cv2.DMatch[]
            List of matches
        task : Task object
            Parent task

        Returns
        -------
        Match[]
            List of matches that pass the ratio test
        """
        s = task.subtask(
            "Ratio",
            desc="Running ratio test").start()
        matches = [
            Match(
                target=self.__kp[m.trainIdx].pt,
                scene=kp[m.queryIdx].pt)
            for m, n in matches
            if m.distance < self.__ratio * n.distance]
        s.done(
            desc="Finished ratio test; {n} points passed"
            .format(n=len(matches)))

        return matches

    def __distance_ratio(self, sampled):
        """Calculate scale factor for a pair of matches.

        Parameters
        ----------
        sampled : Match[2]
            Pair of matches to calculate the ratio between

        Returns
        -------
        MatchPair
            Pair of matches with computed distance; if either keypoint
            is the same, a ratio of -1 is returned.
        """
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
        """Distance ratio analysis to determine target scale in scene

        Parameters
        ----------
        matches : Match[]
            List of matches
        task : Task object
            Parent task

        Returns
        -------
        [Match[], float[]]
            [List of matches, list of scales for histogram]
        """
        s = task.subtask(
            "Scale",
            desc="Pairwise distance ratio test").start()

        # Large number of passing keypoints
        if len(matches) * (len(matches) - 1) / 2 > self.__sample_size:
            s.print("Sampling {n} pairs".format(n=self.__sample_size))
            sampled = [
                self.__distance_ratio(random.sample(matches, 2))
                for _ in range(self.__sample_size)
            ]

        # Small number of passing keypoints
        else:
            s.print("Small number of matches -- using all pairs")
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
        """Compute matches to a target in a given scene

        Parameters
        ----------
        scene : np.array
            Input image
        task : Task object
            Task to use

        Returns
        -------
        [Match[], float[]]
            [List of matches, list of scales for histogram]
        """

        task.start(
            "AKAZE",
            desc="AKAZE Keypoint Matcher searching for " + self.__name)

        kp, matches = self.__search_kp(scene, task)
        matches = self.__ratio_test(kp, matches, task)
        matches, distances = self.__distance_ratio_test(matches, task)

        task.done(desc="Done matching images")

        return matches, distances
